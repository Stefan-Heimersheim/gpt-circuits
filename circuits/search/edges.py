import torch

from circuits import Circuit, Edge, Node
from circuits.search.ablation import Ablator
from circuits.search.divergence import analyze_edge_divergence
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


class EdgeSearch:
    """
    Search for circuit edges in a sparsified model.
    """

    def __init__(self, model: SparsifiedGPT, ablator: Ablator, num_samples: int):
        """
        :param model: The sparsified model to use for circuit extraction.
        :param ablator: Ablation tecnique to use for circuit extraction.
        :param num_samples: The number of samples to use for ablation.
        """
        self.model = model
        self.ablator = ablator
        self.num_samples = num_samples

    def search(
        self,
        tokens: list[int],
        target_token_idx: int,
        upstream_nodes: frozenset[Node],
        downstream_nodes: frozenset[Node],
        threshold: float,
    ) -> frozenset[Edge]:
        """
        Search for circuit edges in a sparsified model.

        :param tokens: The token sequence to use for circuit extraction.
        :param upstream_nodes: The upstream nodes to use for circuit extraction.
        :param downstream_nodes: The downstream nodes to use for circuit extraction.
        :param threshold: The threshold to use for circuit extraction.
        """
        assert len(upstream_nodes) > 0
        upstream_layer_idx = next(iter(upstream_nodes)).layer_idx

        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)

        # Get target logits
        with torch.no_grad():
            output: SparsifiedGPTOutput = self.model(input)
        target_logits = output.logits.squeeze(0)[target_token_idx]  # Shape: (V)

        # Get feature magnitudes
        upstream_magnitudes = output.feature_magnitudes[upstream_layer_idx].squeeze(0)  # Shape: (T, F)
        downstream_magnitudes = output.feature_magnitudes[upstream_layer_idx + 1].squeeze(0)  # Shape: (T, F)

        # Set initial edges as all edges that could exist between upstream and downstream nodes
        initial_edges = set()
        for upstream in sorted(upstream_nodes):
            for downstream in sorted(downstream_nodes):
                if upstream.token_idx <= downstream.token_idx:
                    initial_edges.add(Edge(upstream, downstream))

        # Starting search states
        circuit_edges: frozenset[Edge] = frozenset(initial_edges)  # Circuit to start pruning
        discard_candidates: frozenset[Edge] = frozenset()
        circuit_kl_div: float = float("inf")

        # Start search
        for _ in range(len(initial_edges)):
            # Compute KL divergence
            circuit_candidate = Circuit(
                nodes=upstream_nodes | downstream_nodes,
                edges=frozenset(circuit_edges - discard_candidates),
            )
            circuit_analysis = analyze_edge_divergence(
                self.model,
                self.ablator,
                upstream_layer_idx,
                target_token_idx,
                target_logits,
                [circuit_candidate],
                [upstream_magnitudes],
                [downstream_magnitudes],
                num_samples=self.num_samples,
            )[circuit_candidate]
            circuit_kl_div = circuit_analysis.kl_divergence

            # Print results
            print(
                f"Edges: {len(circuit_candidate.edges)}/{len(initial_edges)} - "
                f"KL Div: {circuit_kl_div:.4f} - "
                f"Predictions: {circuit_analysis.predictions}"
            )

            # If below threshold, continue search
            if circuit_kl_div < threshold:
                # Update circuit
                circuit_edges = circuit_candidate.edges

                # Sort edges by KL divergence (descending)
                estimated_edge_ablation_effects = self.estimate_edge_ablation_effects(
                    upstream_nodes,
                    downstream_nodes,
                    circuit_edges,
                    upstream_magnitudes,
                    downstream_magnitudes,
                    target_token_idx,
                    target_logits,
                )
                least_important_edge = min(estimated_edge_ablation_effects.items(), key=lambda x: x[1])[0]
                least_important_edge_kl_div = estimated_edge_ablation_effects[least_important_edge]
                discard_candidates = frozenset({least_important_edge})

                # Check for early stopping
                if least_important_edge_kl_div > threshold:
                    print("Stopping search - can't improve KL divergence.")
                    break

            # If above threshold, stop search
            else:
                print("Stopping search - KL divergence is too high.")
                break

        # Print final edges (grouped by downstream token)
        print(f"\nCircuit after edge search ({len(circuit_edges)}):")
        for downstream_node in sorted(downstream_nodes):
            edges = [edge for edge in circuit_edges if edge.downstream == downstream_node]
            print(f"{downstream_node}: {', '.join([str(edge.upstream) for edge in sorted(edges)])}")

        return circuit_edges

    def estimate_edge_ablation_effects(
        self,
        upstream_nodes: frozenset[Node],
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        downstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
        target_logits: torch.Tensor,  # Shape: (V)
    ) -> dict[Edge, float]:
        """
        Estimate the KL divergence that results from ablating each edge in a circuit.

        :param upstream_nodes: The upstream nodes to use for estimating ablation effects.
        :param downstream_nodes: The downstream nodes to use for estimating ablation effects.
        :param edges: The edges to use for estimating ablation effects.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param target_token_idx: The target token index.
        :param target_logits: The target logits for the target token.
        """
        # Create a set of circuit variants with one edge removed
        circuit_variants: list[Circuit] = []
        edge_to_circuit_variant: dict[Edge, Circuit] = {}
        for edge in edges:
            circuit_variant = Circuit(upstream_nodes | downstream_nodes, edges=frozenset(edges - {edge}))
            circuit_variants.append(circuit_variant)
            edge_to_circuit_variant[edge] = circuit_variant

        # Calculate KL divergence for each variant
        kld_results = analyze_edge_divergence(
            self.model,
            self.ablator,
            next(iter(upstream_nodes)).layer_idx,
            target_token_idx,
            target_logits,
            circuit_variants,
            [upstream_magnitudes] * len(circuit_variants),
            [downstream_magnitudes] * len(circuit_variants),
            self.num_samples,
        )

        # Map edges to KL divergence
        return {edge: kld_results[variant].kl_divergence for edge, variant in edge_to_circuit_variant.items()}
