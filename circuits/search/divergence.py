from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence

import torch

from circuits import Circuit, Node
from circuits.search.ablation import Ablator
from data.tokenizers import Tokenizer
from models.sparsified import SparsifiedGPT


@dataclass(frozen=True)
class Divergence:
    kl_divergence: float
    predictions: dict[str, float]


@torch.no_grad()
def analyze_node_divergence(
    model: SparsifiedGPT,
    ablator: Ablator,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,  # Shape: (V)
    circuit_variants: Sequence[Circuit],  # List of circuit variants
    feature_magnitudes: Sequence[torch.Tensor],  # List of tensors with shape: (T, F)
    num_samples: int,
) -> dict[Circuit, Divergence]:
    """
    Calculate KL divergence between target logits and logits produced through use of circuit nodes.

    :param model: The sparsified model to use for circuit extraction.
    :param ablator: Ablation tecnique to use for circuit extraction.
    :param layer_idx: The layer index to use for circuit extraction.
    :param target_token_idx: The token index to use for circuit extraction.
    :param target_logits: The target logits for the target token.
    :param circuit_variants: The circuit variants to use for circuit extraction.
    :param feature_magnitudes: A list of feature magnitudes to use for each circuit variant.
    :param num_samples: The number of samples to use for ablation.
    """
    # For storing results
    results: dict[Circuit, Divergence] = {}

    # Patch feature magnitudes for each circuit variant
    patched_feature_magnitudes = patch_feature_magnitudes(
        ablator,
        layer_idx,
        target_token_idx,
        circuit_variants,
        feature_magnitudes,
        num_samples=num_samples,
    )

    # Get predicted logits for each circuit variant when using patched feature magnitudes
    predicted_logits = get_predicted_logits(
        model,
        layer_idx,
        patched_feature_magnitudes,
        target_token_idx,
    )

    # Calculate KL divergence and predictions for each variant
    for circuit_variant, circuit_logits in predicted_logits.items():
        # Compute KL divergence
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(circuit_logits, dim=-1),
            torch.nn.functional.softmax(target_logits, dim=-1),
            reduction="sum",
        )

        # Calculate predictions
        predictions = get_predictions(model.gpt.config.tokenizer, circuit_logits)

        # Store results
        results[circuit_variant] = Divergence(kl_divergence=kl_div.item(), predictions=predictions)

    return results


@torch.no_grad()
def analyze_edge_divergence(
    model: SparsifiedGPT,
    ablator: Ablator,
    upstream_layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,  # Shape: (V)
    circuit_variants: Sequence[Circuit],  # List of circuit variants
    upstream_feature_magnitudes: Sequence[torch.Tensor],  # List of tensors with shape: (T, F)
    downstream_feature_magnitudes: Sequence[torch.Tensor],  # List of tensors with shape: (T, F)
    num_samples: int,
) -> dict[Circuit, Divergence]:
    """
    Calculate KL divergence between target logits and logits produced through use of circuit edges.

    :param model: The sparsified model to use for circuit extraction.
    :param ablator: Ablation tecnique to use for circuit extraction.
    :param upstream_layer_idx: The layer index to use for circuit extraction.
    :param target_token_idx: The token index to use for circuit extraction.
    :param target_logits: The target logits for the target token.
    :param circuit_variants: The circuit variants to use for circuit extraction.
    :param upstream_feature_magnitudes: A list of feature magnitudes to use for each circuit variant.
    :param downstream_feature_magnitudes: A list of feature magnitudes to use for each circuit variant.
    :param num_samples: The number of samples to use for ablation.
    """
    # For storing results
    results: dict[Circuit, Divergence] = {}

    # Patch downstream feature magnitudes using circuit edges
    patched_downstream_magnitudes: list[torch.Tensor] = []  # Shape: (T, F)
    for circuit_variant, upstream_magnitudes, downstream_magnitudes in zip(
        circuit_variants,
        upstream_feature_magnitudes,
        downstream_feature_magnitudes,
    ):
        magnitudes = patch_downstream_feature_magnitudes(
            model,
            upstream_layer_idx,
            upstream_magnitudes,
            downstream_magnitudes,
            circuit_variant,
        )
        patched_downstream_magnitudes.append(magnitudes)

    # Resample feature magnitudes for each circuit variant
    resampled_feature_magnitudes = patch_feature_magnitudes(
        ablator,
        upstream_layer_idx + 1,
        target_token_idx,
        circuit_variants,
        patched_downstream_magnitudes,
        num_samples=num_samples,
    )

    # Get predicted logits for each circuit variant when using patched feature magnitudes
    predicted_logits = get_predicted_logits(
        model,
        upstream_layer_idx + 1,
        resampled_feature_magnitudes,
        target_token_idx,
    )

    # Calculate KL divergence and predictions for each variant
    for circuit_variant, circuit_logits in predicted_logits.items():
        # Compute KL divergence
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(circuit_logits, dim=-1),
            torch.nn.functional.softmax(target_logits, dim=-1),
            reduction="sum",
        )

        # Calculate predictions
        predictions = get_predictions(model.gpt.config.tokenizer, circuit_logits)

        # Store results
        results[circuit_variant] = Divergence(kl_divergence=kl_div.item(), predictions=predictions)

    return results


def patch_feature_magnitudes(
    ablator: Ablator,
    layer_idx: int,
    target_token_idx: int,
    circuit_variants: Sequence[Circuit],
    feature_magnitudes: Sequence[torch.Tensor],
    num_samples: int,
) -> dict[Circuit, torch.Tensor]:  # Shape: (num_samples, T, F)
    """
    Patch feature magnitudes for a list of circuit variants.
    """
    # For mapping variants to patched feature magnitudes
    patched_feature_magnitudes: dict[Circuit, torch.Tensor] = {}

    # Patch feature magnitudes for each variant
    with ThreadPoolExecutor() as executor:
        futures: dict[Future, Circuit] = {}
        for circuit_variant, magnitudes in zip(circuit_variants, feature_magnitudes):
            future = executor.submit(
                ablator.patch,
                layer_idx=layer_idx,
                target_token_idx=target_token_idx,
                feature_magnitudes=magnitudes,
                circuit=circuit_variant,
                num_samples=num_samples,
            )
            futures[future] = circuit_variant

        for future in as_completed(futures):
            circuit_variant = futures[future]
            patched_feature_magnitudes[circuit_variant] = future.result()

    # Return patched feature magnitudes
    return patched_feature_magnitudes


def patch_downstream_feature_magnitudes(
    model: SparsifiedGPT,
    upstream_layer_idx: int,
    upstream_feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    downstream_feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    circuit: Circuit,
) -> torch.Tensor:  # Shape: (T, F)
    """
    Patch downstream feature magnitudes using a set of edges.
    """
    edges = circuit.edges
    upstream_nodes = frozenset({node for node in circuit.nodes if node.layer_idx == upstream_layer_idx})

    # Copy downstream feature magnitudes
    original_downstream_magnitudes = downstream_feature_magnitudes
    downstream_feature_magnitudes = downstream_feature_magnitudes.clone()
    downstream_feature_magnitudes = downstream_feature_magnitudes.squeeze(0)  # Shape: (T, F)

    # Compute downstream feature magnitudes using unpatched upstream feature magnitudes
    computed_downstream_magnitudes = compute_downstream_feature_magnitudes(
        model,
        upstream_layer_idx,
        upstream_feature_magnitudes.unsqueeze(0),
    )

    # Map downstream nodes to upstream dependencies
    downstream_nodes = frozenset([edge.downstream for edge in edges])
    node_to_upstream_dependencies: dict[Node, frozenset[Node]] = {}
    for node in downstream_nodes:
        node_to_upstream_dependencies[node] = frozenset([edge.upstream for edge in edges if edge.downstream == node])
    upstream_dependencies_to_nodes: dict[frozenset[Node], set[Node]] = defaultdict(set)
    for node, dependencies in node_to_upstream_dependencies.items():
        upstream_dependencies_to_nodes[dependencies].add(node)

    # For each downstream node, patch feature magnitudes
    for downstream_node, upstream_dependencies in node_to_upstream_dependencies.items():
        token_idx = downstream_node.token_idx
        feature_idx = downstream_node.feature_idx

        # Find upstream features to ablate
        ablatable_upstream_nodes = set()
        for upstream_node in upstream_nodes - upstream_dependencies:
            if upstream_node.token_idx <= token_idx:
                ablatable_upstream_nodes.add(upstream_node)

        # Ablate upstream features
        ablatable_token_idxs = [node.token_idx for node in ablatable_upstream_nodes]
        ablatable_feature_idxs = [node.feature_idx for node in ablatable_upstream_nodes]
        ablated_upstream_magnitudes = upstream_feature_magnitudes.clone()
        ablated_upstream_magnitudes[ablatable_token_idxs, ablatable_feature_idxs] = 0.0

        # Compute downstream feature magnitudes
        patched_feature_magnitude = compute_downstream_feature_magnitudes(
            model,
            upstream_layer_idx,
            ablated_upstream_magnitudes.unsqueeze(0),
        )[0, token_idx, feature_idx]

        # SAE error that results from using patched upstream feature magnitudes
        sae_error = (
            original_downstream_magnitudes[token_idx, feature_idx]
            - computed_downstream_magnitudes[0, token_idx, feature_idx]
        )

        # Set patched feature magnitude
        downstream_feature_magnitudes[token_idx, feature_idx] = patched_feature_magnitude + sae_error

    return downstream_feature_magnitudes


def compute_downstream_feature_magnitudes(
    model: SparsifiedGPT,
    upstream_layer_idx: int,
    upstream_feature_magnitudes: torch.Tensor,  # Shape: (num_samples, T, F)
) -> torch.Tensor:  # Shape: (num_sample, T, F)
    """
    Compute downstream feature magnitudes via a forward pass through a single transformer block.
    """
    # Reconstruct upstream activations
    x_reconstructed = model.saes[str(upstream_layer_idx)].decode(upstream_feature_magnitudes)  # type: ignore

    # Compute downstream activations
    x_downstream = model.gpt.transformer.h[upstream_layer_idx](x_reconstructed)  # type: ignore

    # Encode to get downstream feature magnitudes
    downstream_sae = model.saes[str(upstream_layer_idx + 1)]
    downstream_feature_magnitudes = downstream_sae(x_downstream).feature_magnitudes
    return downstream_feature_magnitudes


@torch.no_grad()
def get_predicted_logits(
    model: SparsifiedGPT,
    layer_idx: int,
    patched_feature_magnitudes: dict[Circuit, torch.Tensor],  # Shape: (num_samples, T, F)
    target_token_idx: int,
) -> dict[Circuit, torch.Tensor]:  # Shape: (V)
    """
    Get predicted logits for a set of circuit variants when using patched feature magnitudes.

    TODO: Use batching to improve performance
    """
    results: dict[Circuit, torch.Tensor] = {}

    for circuit_variant, feature_magnitudes in patched_feature_magnitudes.items():
        # Reconstruct activations
        x_reconstructed = model.saes[str(layer_idx)].decode(feature_magnitudes)  # type: ignore

        # Compute logits
        predicted_logits = model.gpt.forward_with_patched_activations(
            x_reconstructed, layer_idx=layer_idx
        )  # Shape: (num_samples, T, V)

        # We only care about logits for the target token
        predicted_logits = predicted_logits[:, target_token_idx, :]  # Shape: (num_samples, V)

        # Convert logits to probabilities before averaging across samples
        predicted_probabilities = torch.nn.functional.softmax(predicted_logits, dim=-1)
        predicted_probabilities = predicted_probabilities.mean(dim=0)  # Shape: (V)
        predicted_logits = torch.log(predicted_probabilities)  # Shape: (V)

        # Store results
        results[circuit_variant] = predicted_logits

    return results


def get_predictions(
    tokenizer: Tokenizer,
    logits: torch.Tensor,  # Shape: (V)
    count: int = 5,
) -> dict[str, float]:
    """
    Map logits to probabilities and return top 5 predictions.
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=count)
    results: dict[str, float] = {}
    for i, p in zip(topk.indices, topk.values):
        results[tokenizer.decode_token(int(i.item()))] = round(p.item() * 100, 2)
    return results
