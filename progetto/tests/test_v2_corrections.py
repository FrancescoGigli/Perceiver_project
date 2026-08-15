import math

import pytest
import torch

from src.config.base_cfg import get_base_config
from src.utils.seed import set_global_seed
from src.utils.positional_encoding import FourierPositionalEncoding
from src.perceiver.input_pe import (
    InputPositionalEncoding,
    make_token_permutation,
    pixel_coords,
)
from src.data.cifar10 import CIFAR10PerceiverDataModule
from src.perceiver.attention import CrossAttention, SelfAttention
from src.perceiver.encoder import PerceiverEncoder
from src.perceiver.perceiver import Perceiver
from experiments import EXPERIMENTS


def test_fourier_out_dim_matches_paper_formula():
    """La PE ha d*(2K+1) canali: 2*(2*64+1) = 258 per le immagini."""
    pe = FourierPositionalEncoding(num_bands=64, max_freq=16.0, num_pos_feats=2)
    assert pe.out_dim == 258


def test_fourier_has_no_trainable_parameters():
    """Le Fourier features sono fisse: nessuna proiezione appresa."""
    pe = FourierPositionalEncoding(num_bands=64, max_freq=16.0, num_pos_feats=2)
    assert list(pe.parameters()) == []


def test_fourier_values_match_sin_f_pi_x():
    """Con K=1 e f_max=1 la formula si riduce a [x, sin(pi*x), cos(pi*x)]."""
    pe = FourierPositionalEncoding(num_bands=1, max_freq=1.0, num_pos_feats=2)
    coords = torch.tensor([[0.5, -0.5]])
    out = pe(coords)

    expected = torch.tensor([[
        0.5, -0.5,
        math.sin(math.pi * 0.5), math.sin(math.pi * -0.5),
        math.cos(math.pi * 0.5), math.cos(math.pi * -0.5),
    ]])
    assert out.shape == (1, 6)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


def test_pixel_coords_span_minus_one_to_one():
    coords = pixel_coords(32)
    assert coords.shape == (1024, 2)
    assert coords.min().item() == pytest.approx(-1.0)
    assert coords.max().item() == pytest.approx(1.0)


def test_fourier_input_pe_concatenates_258_channels():
    pe = InputPositionalEncoding(grid_size=32, mode="fourier", num_bands=64, max_freq=16.0)
    assert pe.pe_dim == 258
    assert list(pe.parameters()) == []          # nessun parametro: e' un buffer
    x = torch.zeros(2, 1024, 3)
    assert pe(x).shape == (2, 1024, 261)


def test_learned_input_pe_is_a_trainable_parameter():
    pe = InputPositionalEncoding(grid_size=32, mode="learned", learned_dim=128)
    assert pe.pe_dim == 128
    params = list(pe.parameters())
    assert len(params) == 1
    assert params[0].shape == (1024, 128)
    assert params[0].requires_grad


def test_none_mode_adds_nothing():
    pe = InputPositionalEncoding(grid_size=32, mode="none")
    assert pe.pe_dim == 0
    x = torch.zeros(2, 1024, 3)
    assert pe(x).shape == (2, 1024, 3)


def test_permutation_reorders_tokens_and_keeps_pairs_intact():
    """La permutazione agisce dopo la PE: ogni pixel si porta dietro la sua."""
    perm = make_token_permutation(1024, seed=42)
    plain = InputPositionalEncoding(grid_size=32, mode="fourier", num_bands=4, max_freq=16.0)
    shuffled = InputPositionalEncoding(
        grid_size=32, mode="fourier", num_bands=4, max_freq=16.0, permutation=perm
    )

    x = torch.randn(1, 1024, 3)
    out_plain = plain(x)
    out_shuffled = shuffled(x)

    torch.testing.assert_close(out_shuffled, out_plain.index_select(1, perm))


def test_permutation_is_deterministic_given_the_seed():
    a = make_token_permutation(1024, seed=42)
    b = make_token_permutation(1024, seed=42)
    c = make_token_permutation(1024, seed=7)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_raw_pixels_give_1024_tokens_of_3_channels():
    dm = CIFAR10PerceiverDataModule(patch_size=1)
    assert dm.token_dim == 3
    images = torch.randn(2, 3, 32, 32)
    tokens = dm._to_patches(images)
    assert tokens.shape == (2, 1024, 3)


def test_split_is_disjoint_and_correctly_sized():
    dm = CIFAR10PerceiverDataModule(val_split=5000, split_seed=42)
    dm.setup()
    assert len(dm.train_indices) == 45000
    assert len(dm.val_indices) == 5000
    assert set(dm.train_indices).isdisjoint(set(dm.val_indices))
    assert len(dm.test_dataset) == 10000


def test_split_is_reproducible_from_the_seed():
    a = CIFAR10PerceiverDataModule(val_split=5000, split_seed=42)
    b = CIFAR10PerceiverDataModule(val_split=5000, split_seed=42)
    c = CIFAR10PerceiverDataModule(val_split=5000, split_seed=7)
    a.setup(); b.setup(); c.setup()
    assert a.val_indices == b.val_indices
    assert a.val_indices != c.val_indices


def test_to_patches_is_row_major():
    """Il token j deve venire dal pixel (riga=j//32, col=j%32): l'ordine deve
    combaciare con pixel_coords(), altrimenti ogni pixel prende la PE di un altro."""
    dm = CIFAR10PerceiverDataModule(patch_size=1)
    # Immagine sintetica: il canale 0 codifica l'indice row-major del pixel.
    img = torch.arange(32 * 32, dtype=torch.float).reshape(1, 1, 32, 32)
    img = img.expand(1, 3, 32, 32).contiguous()
    tokens = dm._to_patches(img)
    assert tokens.shape == (1, 1024, 3)
    expected = torch.arange(1024, dtype=torch.float)
    torch.testing.assert_close(tokens[0, :, 0], expected)


def test_validation_uses_test_transform_not_randaugment():
    """La validation split non deve applicare RandAugment: userebbe augmentation
    su dati di selezione. Deve condividere l'oggetto test_transform."""
    dm = CIFAR10PerceiverDataModule(val_split=5000, split_seed=42)
    dm.setup()
    assert dm.val_dataset.dataset.transform is dm.test_transform
    assert dm.train_dataset.dataset.transform is dm.train_transform


def test_val_split_out_of_range_raises():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        CIFAR10PerceiverDataModule(val_split=60000)
    with _pytest.raises(ValueError):
        CIFAR10PerceiverDataModule(val_split=0)


def test_cross_attention_projects_qkv_to_min_of_input_and_latent():
    """Appendice C: Q, K, V hanno min(input_dim, latent_dim) canali."""
    cross = CrossAttention(latent_dim=384, input_dim=261, num_heads=1)
    assert cross.inner_dim == 261
    assert cross.q_proj.out_features == 261
    assert cross.kv_proj.out_features == 522


def test_attention_blocks_use_two_distinct_layernorms():
    """Il LayerNorm prima dell'MLP non e' lo stesso di quello prima dell'attenzione."""
    cross = CrossAttention(latent_dim=384, input_dim=261, num_heads=1)
    assert cross.norm_attn is not cross.norm_ff

    self_attn = SelfAttention(dim=384, num_heads=8)
    assert self_attn.norm_attn is not self_attn.norm_ff


def test_cross_attention_forward_shape():
    cross = CrossAttention(latent_dim=384, input_dim=261, num_heads=1)
    latents = torch.randn(2, 96, 384)
    inputs = torch.randn(2, 1024, 261)
    out = cross(latents, inputs)
    assert out.shape == (2, 96, 384)


def _encoder(**kwargs):
    defaults = dict(
        input_dim=261, latent_dim=384, num_latents=96,
        num_cross_attend_stages=4, num_transformer_blocks=4,
        num_heads_cross=1, num_heads_self=8,
    )
    defaults.update(kwargs)
    return PerceiverEncoder(**defaults)


def test_weight_sharing_gives_L_distinct_latent_blocks():
    """Paper: L blocchi distinti dentro il latent transformer, condivisi fra le T iterazioni."""
    enc = _encoder(weight_sharing=True)
    assert enc.num_distinct_latent_blocks == 4


def test_no_weight_sharing_gives_T_times_L_distinct_latent_blocks():
    enc = _encoder(weight_sharing=False)
    assert enc.num_distinct_latent_blocks == 16


def test_first_cross_attend_has_its_own_weights():
    """Il paper tiene il primo cross-attend separato: condividerlo destabilizzava il training."""
    enc = _encoder(num_cross_attend_stages=4, share_cross_attend=True)
    assert enc.cross_first is not enc.cross_rest
    first = enc.cross_first.q_proj.weight
    rest = enc.cross_rest.q_proj.weight
    assert first.data_ptr() != rest.data_ptr()


def test_single_stage_has_no_shared_cross_attend():
    enc = _encoder(num_cross_attend_stages=1)
    assert enc.cross_rest is None


def test_no_latent_transformer_removes_all_latent_blocks():
    enc = _encoder(use_latent_transformer=False, share_cross_attend=False)
    assert enc.num_distinct_latent_blocks == 0


def test_arrangement_changes_nothing_when_T_is_one():
    latents = torch.randn(1, 96, 384)
    data = torch.randn(1, 32, 261)
    torch.manual_seed(0)
    a = _encoder(num_cross_attend_stages=1, arrangement="interleaved").eval()
    torch.manual_seed(0)
    b = _encoder(num_cross_attend_stages=1, arrangement="at_start").eval()
    with torch.no_grad():
        out_a, _ = a(data, latents)
        out_b, _ = b(data, latents)
    torch.testing.assert_close(out_a, out_b)


def test_encoder_forward_shape():
    enc = _encoder().eval()
    latents = torch.randn(2, 96, 384)
    data = torch.randn(2, 32, 261)
    with torch.no_grad():
        out, maps = enc(data, latents)
    assert out.shape == (2, 96, 384)
    assert maps is None


def test_weight_sharing_reuses_the_same_module_list_object():
    """Non basta il contatore: la stessa ModuleList deve essere fisicamente
    riusata in ogni stage con weight_sharing=True, e oggetti distinti senza."""
    shared = _encoder(weight_sharing=True, num_cross_attend_stages=4)
    assert shared.latent_stages[0] is shared.latent_stages[3]

    distinct = _encoder(weight_sharing=False, num_cross_attend_stages=4)
    assert distinct.latent_stages[0] is not distinct.latent_stages[3]


def test_no_weight_sharing_has_more_parameters_than_sharing():
    """La differenza di semantica deve riflettersi nei parametri reali."""
    shared = _encoder(weight_sharing=True)
    distinct = _encoder(weight_sharing=False)
    ps = sum(p.numel() for p in shared.parameters())
    pd = sum(p.numel() for p in distinct.parameters())
    assert pd > ps


def test_at_start_differs_from_interleaved_when_T_is_greater_than_one():
    """Con T>1 le due disposizioni NON devono essere equivalenti: at_start
    applica tutti i cross-attend prima, interleaved li alterna ai blocchi latenti.
    Stesso seed -> stessi pesi -> output diversi se l'ordine conta davvero."""
    latents = torch.randn(1, 96, 384)
    data = torch.randn(1, 32, 261)
    torch.manual_seed(0)
    inter = _encoder(num_cross_attend_stages=2, arrangement="interleaved").eval()
    torch.manual_seed(0)
    at_start = _encoder(num_cross_attend_stages=2, arrangement="at_start").eval()
    with torch.no_grad():
        out_inter, _ = inter(data, latents)
        out_at_start, _ = at_start(data, latents)
    assert not torch.allclose(out_inter, out_at_start, atol=1e-5)


def _model(**kwargs):
    pe = kwargs.pop("input_pe", None) or InputPositionalEncoding(
        grid_size=8, mode="fourier", num_bands=4, max_freq=4.0
    )
    defaults = dict(
        token_dim=3, num_classes=10, input_pe=pe,
        num_latents=16, latent_dim=32, num_cross_attend_stages=2,
        num_transformer_blocks=2, num_heads_cross=1, num_heads_self=4,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return Perceiver(**defaults)


def test_latents_use_truncated_normal_with_small_scale():
    model = _model(num_latents=4096, latent_dim=64, latent_init_scale=0.02)
    std = model.latents.std().item()
    assert 0.015 < std < 0.025          # ~0.02, non ~1.0 come in v1
    assert model.latents.abs().max().item() <= 2.0 * 0.02 + 1e-6


def test_classifier_is_a_bare_linear_layer():
    """Il paper: media sui latenti, poi un solo Linear. Nessuna LayerNorm in mezzo."""
    model = _model()
    assert isinstance(model.classifier, torch.nn.Linear)


def test_attention_is_permutation_invariant_without_pe():
    """Senza PE l'output non dipende dall'ordine dei token.

    Se questo test fallisce, il modello non e' un Perceiver: il cross-attention
    e' una somma pesata, e una somma non ha ordine.
    """
    model_no_pe = _model(input_pe=InputPositionalEncoding(grid_size=8, mode="none")).eval()
    x = torch.randn(2, 64, 3)
    perm = make_token_permutation(64, seed=1)

    with torch.no_grad():
        plain = model_no_pe(x)
        shuffled = model_no_pe(x.index_select(1, perm))

    torch.testing.assert_close(plain, shuffled, atol=1e-4, rtol=1e-4)


def test_permuted_pe_leaves_the_output_unchanged():
    """Permutare (pixel, PE) insieme non cambia l'output: e' l'esperimento della Tab. 2."""
    perm = make_token_permutation(64, seed=1)
    torch.manual_seed(0)
    plain = _model(input_pe=InputPositionalEncoding(grid_size=8, mode="fourier", num_bands=4, max_freq=4.0)).eval()
    torch.manual_seed(0)
    shuffled = _model(
        input_pe=InputPositionalEncoding(
            grid_size=8, mode="fourier", num_bands=4, max_freq=4.0, permutation=perm
        )
    ).eval()

    x = torch.randn(2, 64, 3)
    with torch.no_grad():
        a = plain(x)
        # NB: si passa x "grezzo" a entrambi i modelli. E' il modulo PE del
        # modello "shuffled" (il suo self.perm interno) ad applicare la
        # permutazione a (pixel, PE) insieme, come gia' validato da
        # test_permutation_reorders_tokens_and_keeps_pairs_intact in questo
        # stesso file. Pre-permutare x qui (x.index_select(1, perm)) applica
        # la permutazione due volte e rompe il test: verificato numericamente
        # (diff ~0.05 con il doppio, ~1e-7 con x grezzo). Vedi task-6-report.md.
        b = shuffled(x)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_same_seed_gives_identical_initial_weights():
    set_global_seed(42)
    a = _model()
    set_global_seed(42)
    b = _model()
    set_global_seed(7)
    c = _model()

    torch.testing.assert_close(a.latents, b.latents, atol=0, rtol=0)
    assert not torch.allclose(a.latents, c.latents)


def test_defaults_match_the_paper_faithful_base_config():
    args = get_base_config().parse_args([])
    assert args.seed == 42
    assert args.patch_size == 1
    assert args.val_split == 5000
    assert args.fourier_num_bands == 64
    assert args.fourier_max_freq == 16.0
    assert args.latent_init_scale == 0.02
    assert args.num_heads_cross == 1
    assert args.num_heads_self == 8
    assert args.cross_attend_arrangement == "interleaved"
    assert args.dropout == 0.0
    assert not hasattr(args, "cifar10_fourier_bands")


def test_registry_has_42_runs_with_unique_ids():
    ids = [e["id"] for e in EXPERIMENTS]
    # Perceiver: 24 immagini (con il controllo senza PE) + 3 point cloud.
    # Perceiver IO: 2 immagini + 1 MLM + 10 GLUE + 1 multitask. Piu' 1 baseline CNN.
    assert len(ids) == 42
    assert len(set(ids)) == 42
    counts = {}
    for e in EXPERIMENTS:
        m = e.get("modality", "image")
        counts[m] = counts.get(m, 0) + 1
    assert counts == {"image": 24, "modelnet": 3, "io_image": 2, "io_mlm": 1,
                      "io_glue": 10, "io_multitask": 1, "baseline": 1}


def test_io_cifar_uses_the_paper_recipe_not_the_v1_multistep():
    """Il paper IO (App. A.1) abbandona il multistep del Perceiver v1 per la
    classificazione: lr basso + cosine. Con la ricetta v1 (4e-3 multistep) la run
    IO su CIFAR sotto-allena (train fermo, val oscillante): questo test impedisce
    di reintrodurla per sbaglio, e verifica che TUTTO il resto resti gemello di e01."""
    from experiments import command_for

    def effective(cmd):
        # argparse tiene l'ultima occorrenza: e' la semantica del registro
        return {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}

    io01 = effective(command_for(next(e for e in EXPERIMENTS if e["id"] == "io01_cifar")))
    e01 = effective(command_for(next(e for e in EXPERIMENTS if e["id"] == "e01_baseline")))

    assert io01["--lr"] == "0.001" and io01["--scheduler"] == "cosine"
    # architettura e dati identici a e01: stesse chiavi condivise, stessi valori
    for key in ("--num_latents", "--latent_dim", "--num_cross_attend_stages",
                "--num_transformer_blocks", "--fourier_num_bands", "--fourier_max_freq",
                "--latent_init_scale", "--dropout", "--epochs", "--batch_size_cifar10",
                "--patch_size", "--val_split", "--optimizer"):
        assert io01[key] == e01[key], f"{key}: {io01[key]} != {e01[key]}"


def test_glue_finetuning_matches_the_mlm_pretraining_sequence_length():
    """I positional encoding si trasferiscono solo se seq_len coincide fra MLM e GLUE."""
    from experiments import BASE_IO_GLUE, BASE_IO_MLM

    def seq_len(base):
        return base[base.index("--text_seq_len") + 1]

    assert seq_len(BASE_IO_MLM) == seq_len(BASE_IO_GLUE)


def test_glue_and_mlm_build_the_same_input_representation():
    """Il transfer MLM -> GLUE vale solo se l'input ha la STESSA larghezza e lo STESSO
    positional encoding: con due Fourier diversi il primo cross-attention del
    checkpoint viene scartato e il fine-tuning riparte da pesi casuali proprio nel
    modulo che legge il testo (seq_len uguale non basta a garantirlo)."""
    from src.data.glue_sst2 import SST2PerceiverDataModule
    from src.data.glue_tasks import GLUEPerceiverDataModule
    from src.data.wikitext103 import WikiText103PerceiverDataModule

    kw = dict(seq_len=64, fourier_dim=64, max_frequencies=64.0)
    mlm = WikiText103PerceiverDataModule(data_dir="./data", num_frequency_bands=6, **kw)
    glue = GLUEPerceiverDataModule(task_name="rte", data_dir="./data", **kw)
    sst2 = SST2PerceiverDataModule(data_dir="./data", **kw)

    assert mlm.input_dim == glue.input_dim == sst2.input_dim
    assert torch.allclose(mlm.pos_encodings, glue.pos_encodings)
    assert torch.allclose(mlm.pos_encodings, sst2.pos_encodings)


def test_perceiver_io_latents_use_the_configured_init_scale():
    """Con torch.randn i latenti partivano a std 1.0, cioe' la configurazione che in
    Fig. 6 fa collassare il training (e28_init_scale_1p0)."""
    from src.perceiver_io.perceiver_io import PerceiverIO

    model = PerceiverIO(input_dim=261, num_classes=10, num_latents=96, latent_dim=384,
                        latent_init_scale=0.02)
    assert model.latents.detach().std() < 0.05
    assert model.output_queries.detach().std() < 0.05


def test_perceiver_io_applies_the_input_positional_encoding_on_images():
    """Senza input_pe il modello IO riceveva 3 canali grezzi pur essendo costruito
    per token+PE: errore di shape al primo batch."""
    from src.perceiver.input_pe import InputPositionalEncoding
    from src.perceiver_io.perceiver_io import PerceiverIO

    pe = InputPositionalEncoding(grid_size=8, mode="fourier", num_bands=4, max_freq=4.0)
    model = PerceiverIO(input_dim=3 + pe.pe_dim, num_classes=10, num_latents=8,
                        latent_dim=32, num_heads=4, input_pe=pe)
    raw_tokens = torch.randn(2, 64, 3)          # come li produce il DataModule CIFAR
    assert model(raw_tokens).shape == (2, 10)


def test_io_summary_computes_the_glue_average_and_the_pretraining_delta(capsys):
    """La media GLUE e' il numero confrontabile con la Tab. 1 del paper; il delta
    pretrained-vs-scratch e' cio' che i due controlli devono misurare."""
    from analyze_v2 import print_io_section

    results = {
        "e01_baseline": {"test_accuracy": 0.70},
        "io01_cifar": {"test_accuracy": 0.72},
        "io_mlm": {"val_accuracy": 0.80},
        # media voluta: (0.60+0.80+0.70+0.50+0.60+0.40+0.70+0.50)/8 = 0.6
        "io_glue_cola": {"val_accuracy": 0.60}, "io_glue_sst2": {"val_accuracy": 0.80},
        "io_glue_mrpc": {"val_accuracy": 0.70}, "io_glue_stsb": {"val_accuracy": 0.50},
        "io_glue_qqp": {"val_accuracy": 0.60}, "io_glue_mnli": {"val_accuracy": 0.40},
        "io_glue_qnli": {"val_accuracy": 0.70}, "io_glue_rte": {"val_accuracy": 0.50},
        "io_glue_sst2_scratch": {"val_accuracy": 0.65},
        "io_glue_rte_scratch": {"val_accuracy": 0.55},
    }
    print_io_section(results, band=0.0278)
    out = capsys.readouterr().out

    assert "media su 8/8 task: 60.00" in out
    assert "ATTENZIONE: media parziale" not in out
    assert "+15.00%" in out          # sst2: 0.80 col pre-training contro 0.65 da zero
    assert "-5.00%" in out           # rte: 0.50 contro 0.55, il pre-training non aiuta
    assert "+2.00%" in out           # io01_cifar vs e01_baseline
    assert "NON concludente" in out  # 2 punti stanno dentro la banda di rumore (2.78)


def test_a_run_with_a_valid_result_is_never_queued_for_rerun(tmp_path, monkeypatch):
    """e28_init_scale_1p0 finisce sotto il livello del caso: e' il risultato della
    Fig. 6, non un errore. Il numero riportato viene dal checkpoint scelto sulla
    validation, quindi la run e' completa; rilanciarla con lo stesso seed darebbe lo
    stesso esito e --next ci resterebbe dentro in loop."""
    import json as _json

    import experiments as _exp

    run_dir = tmp_path / "logs" / "e28_init_scale_1p0"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(_json.dumps({
        "test_accuracy": 0.5208, "val_accuracy": 0.5336,
        "final_val_accuracy": 0.4632, "selected_epoch": 9,
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exp = next(e for e in EXPERIMENTS if e["id"] == "e28_init_scale_1p0")
    stato, acc, pulito = _exp._run_status(exp)
    assert pulito is True, "una run con un risultato valido non va rimessa in coda"
    assert "instabile" in stato
    assert acc == pytest.approx(0.5208)


def test_a_run_without_any_result_is_queued_for_rerun(tmp_path, monkeypatch):
    import json as _json

    import experiments as _exp

    run_dir = tmp_path / "logs" / "e28_init_scale_1p0"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(_json.dumps({
        "test_accuracy": None, "val_accuracy": None,
        "final_val_accuracy": None, "selected_epoch": 3,
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exp = next(e for e in EXPERIMENTS if e["id"] == "e28_init_scale_1p0")
    stato, _, pulito = _exp._run_status(exp)
    assert pulito is False and "rifare" in stato


def test_glue_runs_are_not_flagged_as_diverged_by_the_cifar_threshold():
    """La soglia 0.5 vale per CIFAR-10 (caso 0.10), non per GLUE: i task binari hanno
    il caso proprio a 0.50 e STS-B e' una regressione con valori negativi. Con la
    soglia sbagliata ogni run GLUE risultava divergita e --next restava in loop."""
    from experiments import _COLLAPSE_FLOOR

    assert _COLLAPSE_FLOOR["io_glue"] is None
    assert _COLLAPSE_FLOOR["image"] == 0.5


def test_wikitext2_does_not_redownload_when_the_data_is_already_extracted(tmp_path, monkeypatch):
    """Il controllo 'gia' scaricato' deve guardare nella cartella annidata in cui lo
    zip estrae (e da cui legge setup()), altrimenti riscarica a ogni run."""
    import urllib.request

    from src.data.wikitext2 import WikiText2PerceiverDataModule

    extracted = tmp_path / "wikitext-2" / "wikitext-2"
    extracted.mkdir(parents=True)
    for name in ("wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens"):
        (extracted / name).write_text("ciao\n", encoding="utf-8")

    def boom(*args, **kwargs):
        raise AssertionError("ha tentato il download benche' i dati fossero gia' estratti")

    monkeypatch.setattr(urllib.request, "urlretrieve", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)

    dm = WikiText2PerceiverDataModule(data_dir=str(tmp_path), seq_len=8)
    dm._download_wikitext2()  # non deve fare rete


def test_short_runs_do_not_use_multistep_whose_milestones_are_fixed_at_120_epochs():
    """MultiStepLR ha milestone fisse [84,102,114]: su una run piu' corta il
    learning rate non decadrebbe mai. Le run IO (10/30 epoche) devono usare cosine."""
    from experiments import command_for

    for exp in EXPERIMENTS:
        cmd = command_for(exp)
        if "--scheduler" not in cmd:
            continue
        epochs = int(cmd[cmd.index("--epochs") + 1])
        scheduler = cmd[cmd.index("--scheduler") + 1]
        if scheduler == "multistep":
            assert epochs > 114, f"{exp['id']}: multistep con {epochs} epoche non decade mai"


def test_glue_runs_load_the_mlm_checkpoint_except_the_scratch_controls():
    glue = [e for e in EXPERIMENTS if e.get("modality") == "io_glue"]
    pretrained = [e for e in glue if not e["id"].endswith("_scratch")]
    scratch = [e for e in glue if e["id"].endswith("_scratch")]
    assert len(pretrained) == 8 and len(scratch) == 2
    for exp in pretrained:
        assert "--load_checkpoint_path" in exp["overrides"]
    for exp in scratch:
        assert "--load_checkpoint_path" not in exp["overrides"]


def test_every_override_is_a_known_flag():
    parser = get_base_config()
    known = {action.option_strings[0] for action in parser._actions if action.option_strings}
    for exp in EXPERIMENTS:
        for token in exp["overrides"]:
            if token.startswith("--"):
                assert token in known, f"{exp['id']}: flag sconosciuto {token}"


def test_tab5_runs_do_not_share_cross_attends():
    """Il paper: 'we do not share weights between cross-attention modules' (Tab. 5)."""
    tab5 = [e for e in EXPERIMENTS if e["group"] == "tab5"]
    assert len(tab5) == 3
    for exp in tab5:
        assert "--no_latent_transformer" in exp["overrides"]
        assert "--no_share_cross_attend" in exp["overrides"]


def test_perceiver_io_constructs_with_text_input_dims():
    """PerceiverIO deve costruirsi per larghezze d'input tipiche del testo,
    dove input_dim non e' divisibile per num_heads (regressione: il cross-attn
    a min(C,D) richiedeva divisibilita', num_heads_cross=1 la garantisce)."""
    from src.perceiver_io.perceiver_io import PerceiverIO
    for input_dim in (270, 257, 261):
        model = PerceiverIO(
            input_dim=input_dim, num_classes=2,
            num_latents=64, latent_dim=256, num_heads=4,
        )
        assert model is not None
