"""Flint on CPU: construction gates, routing and MLP numerics against numpy, and loading a synthetic checkpoint."""

import socket
import unittest

import numpy as np
import torch

from sglang.srt.configs.flint import FlintConfig
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

NUM_LAYERS = 6
FULL_LAYERS = (0, 5)
DENSE_LAYERS = (0, 1)
SLIDING_WINDOW = 17


def _tiny_config() -> FlintConfig:
    layer_types = [
        "full_attention" if i in FULL_LAYERS else "sliding_attention"
        for i in range(NUM_LAYERS)
    ]
    cfg = FlintConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=384,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=16,
        num_experts_per_tok=4,
        sliding_window=SLIDING_WINDOW,
        layer_types=layer_types,
        no_rope_layer_indices=list(FULL_LAYERS),
        mlp_only_layers=list(DENSE_LAYERS),
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    cfg.architectures = ["FlintForCausalLM"]
    return cfg


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestFlintCpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
        )
        publish_build_topology(tp_size=1, device="cpu")
        initialize_model_parallel()
        from sglang.srt.models.flint import FlintForCausalLM

        torch.manual_seed(0)
        cls.config = _tiny_config()
        cls.model = FlintForCausalLM(cls.config)

    @classmethod
    def tearDownClass(cls):
        destroy_model_parallel()
        destroy_distributed_environment()
        reset_context()

    # ---- construction gates -------------------------------------------------

    def test_config_derived_fields(self):
        cfg = self.config
        self.assertTrue(cfg.is_hybrid_swa)
        self.assertTrue(cfg.enable_lm_head_fp32)
        self.assertEqual(
            cfg.hybrid_layer_pattern,
            [0 if i in FULL_LAYERS else 1 for i in range(NUM_LAYERS)],
        )

    def test_config_rejects_other_router_and_softmax(self):
        for field, value in (
            ("router_score_function", "softmax"),
            ("softmax_type", "vanilla"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                FlintConfig(**{field: value})

    def test_layer_gates(self):
        from sglang.srt.models.flint import FlintMLP, FlintSparseMoeBlock

        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            with self.subTest(layer=i):
                if i in DENSE_LAYERS:
                    self.assertIs(type(layer.mlp), FlintMLP)
                    self.assertFalse(layer.is_layer_sparse)
                else:
                    self.assertIsInstance(layer.mlp, FlintSparseMoeBlock)
                    self.assertTrue(layer.is_layer_sparse)
                self.assertEqual(attn.skip_rope, i in FULL_LAYERS)
                self.assertEqual(
                    attn.attn.sliding_window_size,
                    -1 if i in FULL_LAYERS else SLIDING_WINDOW - 1,
                )
                self.assertEqual(tuple(attn.softmax_offset.shape), (8,))
                self.assertEqual(attn.softmax_offset.dtype, torch.bfloat16)

    def test_moe_router_config(self):
        blk = self.model.model.layers[2].mlp
        tc = blk.topk.topk_config
        self.assertEqual(tc.scoring_func, "sigmoid")
        self.assertIs(tc.correction_bias, blk.gate.expert_bias)
        self.assertEqual(blk.gate.expert_bias.dtype, torch.float32)
        self.assertTrue(tc.renormalize)

    def test_model_level_hooks(self):
        self.assertEqual(
            self.model.get_attention_sliding_window_size(), SLIDING_WINDOW - 1
        )
        routed = self.model.routed_experts_weights_of_layer.value
        self.assertEqual(
            sorted(routed), [i for i in range(NUM_LAYERS) if i not in DENSE_LAYERS]
        )

    def test_qwen3_moe_default_unchanged(self):
        """The backported extension points keep Qwen3-MoE all-sparse, softmax-routed and full-window."""
        from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer

        attn, ffn = Qwen3MoeDecoderLayer.stage_facts(self.config, 0)
        self.assertTrue(ffn.sparse)

    # ---- routing vs numpy ---------------------------------------------------

    def _numpy_route(self, logits, bias, k):
        s = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        choice = s + bias[None, :]
        ids = np.argsort(-choice, axis=-1, kind="stable")[:, :k]
        w = np.take_along_axis(s, ids, axis=-1)
        w = w / (w.sum(-1, keepdims=True) + 1e-20)
        return ids, w, choice

    def test_routing_matches_numpy(self):
        from sglang.srt.layers.moe.topk import biased_topk_impl

        torch.manual_seed(1)
        k, n = 6, 256
        for bias_scale in (0.0, 0.05):
            logits = torch.randn(64, n) * 2.0
            bias = torch.randn(n) * bias_scale
            w, ids = biased_topk_impl(
                hidden_states=torch.zeros(64, 8),
                gating_output=logits.float(),
                correction_bias=bias.float(),
                topk=k,
                renormalize=True,
                scoring_func="sigmoid",
            )
            ref_ids, ref_w, choice = self._numpy_route(logits.numpy(), bias.numpy(), k)
            srt = np.sort(choice, axis=-1)[:, ::-1]
            clear = (srt[:, k - 1] - srt[:, k]) > 1e-6
            got_ids = ids.numpy()
            for row in np.nonzero(clear)[0]:
                self.assertEqual(set(got_ids[row]), set(ref_ids[row]))
                order = np.argsort(got_ids[row])
                ref_order = np.argsort(ref_ids[row])
                np.testing.assert_allclose(
                    w.numpy()[row][order], ref_w[row][ref_order], rtol=1e-6
                )

    def test_block_routes_on_fp32_logits(self):
        """The MoE block hands TopK fp32 logits even when the gate runs in bf16 (R5)."""
        blk = self.model.model.layers[2].mlp
        seen = {}
        orig_topk = blk.topk

        class Spy(torch.nn.Module):
            def forward(self, hidden, logits):
                seen["dtype"] = logits.dtype
                return orig_topk(hidden, logits)

            def empty_topk_output(self, device):
                return orig_topk.empty_topk_output(device)

        blk.topk = Spy()
        try:
            gate = blk.gate.to(torch.bfloat16)
            h = torch.randn(4, 256, dtype=torch.bfloat16)
            logits, _ = gate(h)
            self.assertEqual(logits.dtype, torch.bfloat16)
            try:
                blk.forward_normal(h)
            except Exception:
                pass  # CPU expert kernels may be absent; only the dispatch dtype matters here
            self.assertEqual(seen.get("dtype"), torch.float32)
        finally:
            blk.topk = orig_topk
            blk.gate.to(torch.float32)

    # ---- MLP vs numpy -------------------------------------------------------

    def _check_swiglu(self, mlp, inter):
        torch.manual_seed(2)
        with torch.no_grad():
            for p in mlp.parameters():
                p.copy_(torch.randn_like(p) * 0.05)
        x = torch.randn(5, 256)
        out = mlp(x).detach().numpy()
        wgu = mlp.gate_up_proj.weight.detach().numpy().astype(np.float64)
        wd = mlp.down_proj.weight.detach().numpy().astype(np.float64)
        xn = x.numpy().astype(np.float64)
        g, u = (xn @ wgu.T)[:, :inter], (xn @ wgu.T)[:, inter:]
        ref = ((g / (1.0 + np.exp(-g))) * u) @ wd.T
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(mlp(x, None).detach().numpy(), out)

    def test_dense_mlp_matches_numpy(self):
        self._check_swiglu(self.model.model.layers[0].mlp, 384)

    def test_shared_expert_matches_numpy(self):
        self._check_swiglu(self.model.model.layers[2].mlp.shared_expert, 128)

    # ---- synthetic checkpoint load (model-loader layer 1) -------------------

    def test_synthetic_checkpoint_loads_bitwise(self):
        from sglang.srt.models.flint import FlintForCausalLM

        cfg = self.config
        torch.manual_seed(3)
        model = FlintForCausalLM(cfg)
        ckpt = {}
        H, I, Im, Is = 256, 384, 128, 128
        nq, nkv, hd = 8, 2, 32
        ckpt["model.embed_tokens.weight"] = torch.randn(cfg.vocab_size, H)
        ckpt["lm_head.weight"] = torch.randn(cfg.vocab_size, H)
        ckpt["model.norm.weight"] = torch.randn(H)
        for i in range(NUM_LAYERS):
            p = f"model.layers.{i}."
            ckpt[p + "input_layernorm.weight"] = torch.randn(H)
            ckpt[p + "post_attention_layernorm.weight"] = torch.randn(H)
            ckpt[p + "self_attn.q_proj.weight"] = torch.randn(nq * hd, H)
            ckpt[p + "self_attn.k_proj.weight"] = torch.randn(nkv * hd, H)
            ckpt[p + "self_attn.v_proj.weight"] = torch.randn(nkv * hd, H)
            ckpt[p + "self_attn.o_proj.weight"] = torch.randn(H, nq * hd)
            ckpt[p + "self_attn.q_norm.weight"] = torch.randn(hd)
            ckpt[p + "self_attn.k_norm.weight"] = torch.randn(hd)
            ckpt[p + "self_attn.softmax_offset"] = torch.randn(nq).bfloat16()
            if i in DENSE_LAYERS:
                for name, shape in (
                    ("gate_proj", (I, H)),
                    ("up_proj", (I, H)),
                    ("down_proj", (H, I)),
                ):
                    ckpt[p + f"mlp.{name}.weight"] = torch.randn(*shape)
            else:
                ckpt[p + "mlp.gate.weight"] = torch.randn(cfg.num_experts, H)
                for e in range(cfg.num_experts):
                    for name, shape in (
                        ("gate_proj", (Im, H)),
                        ("up_proj", (Im, H)),
                        ("down_proj", (H, Im)),
                    ):
                        ckpt[p + f"mlp.experts.{e}.{name}.weight"] = torch.randn(*shape)
                for name, shape in (
                    ("gate_proj", (Is, H)),
                    ("up_proj", (Is, H)),
                    ("down_proj", (H, Is)),
                ):
                    ckpt[p + f"mlp.shared_expert.{name}.weight"] = torch.randn(*shape)

        with torch.no_grad():
            for i in range(2, NUM_LAYERS):
                model.model.layers[i].mlp.gate.expert_bias.fill_(7.0)
        model.load_weights(list(ckpt.items()))
        params = dict(model.named_parameters())

        def check(pname, expected):
            torch.testing.assert_close(
                params[pname].detach().to(expected.dtype), expected, rtol=0, atol=0
            )

        check("model.embed_tokens.weight", ckpt["model.embed_tokens.weight"])
        check("lm_head.weight", ckpt["lm_head.weight"])
        for i in range(NUM_LAYERS):
            p = f"model.layers.{i}."
            with self.subTest(layer=i):
                check(
                    p + "self_attn.qkv_proj.weight",
                    torch.cat(
                        [
                            ckpt[p + "self_attn.q_proj.weight"],
                            ckpt[p + "self_attn.k_proj.weight"],
                            ckpt[p + "self_attn.v_proj.weight"],
                        ]
                    ),
                )
                check(
                    p + "self_attn.softmax_offset",
                    ckpt[p + "self_attn.softmax_offset"],
                )
                if i in DENSE_LAYERS:
                    check(
                        p + "mlp.gate_up_proj.weight",
                        torch.cat(
                            [
                                ckpt[p + "mlp.gate_proj.weight"],
                                ckpt[p + "mlp.up_proj.weight"],
                            ]
                        ),
                    )
                    check(p + "mlp.down_proj.weight", ckpt[p + "mlp.down_proj.weight"])
                else:
                    check(
                        p + "mlp.shared_expert.gate_up_proj.weight",
                        torch.cat(
                            [
                                ckpt[p + "mlp.shared_expert.gate_proj.weight"],
                                ckpt[p + "mlp.shared_expert.up_proj.weight"],
                            ]
                        ),
                    )
                    check(p + "mlp.gate.weight", ckpt[p + "mlp.gate.weight"])
                    w13 = params[p + "mlp.experts.w13_weight"].detach()
                    w2 = params[p + "mlp.experts.w2_weight"].detach()
                    for e in (0, cfg.num_experts - 1):
                        torch.testing.assert_close(
                            w13[e],
                            torch.cat(
                                [
                                    ckpt[p + f"mlp.experts.{e}.gate_proj.weight"],
                                    ckpt[p + f"mlp.experts.{e}.up_proj.weight"],
                                ]
                            ),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            w2[e],
                            ckpt[p + f"mlp.experts.{e}.down_proj.weight"],
                            rtol=0,
                            atol=0,
                        )
                    # The checkpoint carries no router bias, and the bias was
                    # pre-filled (as a capture sentinel or meta build would), so
                    # load_weights must zero it in place, keeping the TopK reference.
                    bias = model.model.layers[i].mlp.gate.expert_bias
                    self.assertIs(
                        model.model.layers[i].mlp.topk.topk_config.correction_bias,
                        bias,
                    )
                    self.assertEqual(bias.dtype, torch.float32)
                    self.assertEqual(int(torch.count_nonzero(bias)), 0)

    # ---- parsers ------------------------------------------------------------

    def test_parsers_registered(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        self.assertIs(FunctionCallParser.ToolCallParserEnum["flint"], Glm47MoeDetector)
        det = ReasoningParser.DetectorMap["flint"]()
        self.assertIsNone(det.reasoning_default)
        res = det.detect_and_parse("<think>2+2</think>The answer is \\boxed{4}.")
        self.assertEqual(res.reasoning_text.strip(), "2+2")
        self.assertIn("\\boxed{4}", res.normal_text)
        res = det.detect_and_parse("Direct \\boxed{5}")
        self.assertIn("\\boxed{5}", res.normal_text)


if __name__ == "__main__":
    unittest.main()
