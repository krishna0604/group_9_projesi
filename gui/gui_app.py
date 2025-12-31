import customtkinter as ctk
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

random.seed(42)
np.random.seed(42)


def create_erdos_renyi_connected(n=250, p=0.4, seed=42):
    """Generate connected Erdős-Rényi graph with node/edge attributes as placeholders."""
    fixed_seed = seed
    while True:
        G = nx.erdos_renyi_graph(n, p, seed=fixed_seed)
        if nx.is_connected(G):
            break
        fixed_seed += 1

    # Node attributes
    for node in G.nodes():
        G.nodes[node]['ProcessingDelay'] = random.uniform(0.5, 2.0)
        G.nodes[node]['NodeReliability'] = random.uniform(0.95, 0.999)

    # Edge attributes
    for u, v in G.edges():
        G.edges[u, v]['Bandwidth'] = random.uniform(100, 1000)
        G.edges[u, v]['LinkDelay'] = random.uniform(3, 15)
        G.edges[u, v]['LinkReliability'] = random.uniform(0.95, 0.999)

    return G


def compute_path_metrics(graph, path):
    if not path:
        return 0.0, float('inf'), float('inf')

    total_reliability = 1.0
    total_delay = 0.0
    resource_cost = 0.0  # sum(1/bw)

    for node in path:
        total_delay += graph.nodes[node].get('ProcessingDelay', 0.0)
        total_reliability *= graph.nodes[node].get('NodeReliability', 1.0)

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = graph.edges.get((u, v), {})
        total_delay += edge.get('LinkDelay', 0.0)
        bw = edge.get('Bandwidth', 0.0)
        if bw > 0:
            resource_cost += (1.0 / bw)
        else:
            resource_cost = float('inf')

    return total_delay, total_reliability, resource_cost


class RoutingGUI(ctk.CTk):
    def __init__(self, graph=None):
        super().__init__()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.title("Network Routing Optimization — GUI")
        self.geometry("1200x720")

        self.graph = graph if graph is not None else create_erdos_renyi_connected()
        self.pos = nx.spring_layout(self.graph, seed=42)

        # Left: plotting canvas
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.place(relx=0.01, rely=0.01, relwidth=0.68, relheight=0.98)

        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Right: controls
        self.ctrl_frame = ctk.CTkFrame(self)
        self.ctrl_frame.place(relx=0.70, rely=0.01, relwidth=0.29, relheight=0.98)

        # Source/Dest
        self.s_label = ctk.CTkLabel(self.ctrl_frame, text="Source Node (S):")
        self.s_label.pack(pady=(12, 0))
        self.s_var = ctk.CTkEntry(self.ctrl_frame, width=80)
        self.s_var.insert(0, "0")
        self.s_var.pack(pady=(6, 12))

        self.d_label = ctk.CTkLabel(self.ctrl_frame, text="Destination Node (D):")
        self.d_label.pack(pady=(6, 0))
        self.d_var = ctk.CTkEntry(self.ctrl_frame, width=80)
        self.d_var.insert(0, str(len(self.graph.nodes()) - 1))
        self.d_var.pack(pady=(6, 12))

        # Weight sliders
        self.w_label = ctk.CTkLabel(self.ctrl_frame, text="Optimization weights (sum normalized to 1.0)")
        self.w_label.pack(pady=(8, 6))

        self.w_delay = ctk.CTkSlider(self.ctrl_frame, from_=0.0, to=1.0, number_of_steps=100)
        self.w_delay.set(0.33)
        self.w_delay.pack(pady=6, padx=12)
        self.w_delay_label = ctk.CTkLabel(self.ctrl_frame, text="W_delay: 0.33")
        self.w_delay_label.pack()

        self.w_reliability = ctk.CTkSlider(self.ctrl_frame, from_=0.0, to=1.0, number_of_steps=100)
        self.w_reliability.set(0.33)
        self.w_reliability.pack(pady=6, padx=12)
        self.w_reliability_label = ctk.CTkLabel(self.ctrl_frame, text="W_reliability: 0.33")
        self.w_reliability_label.pack()

        self.w_resource = ctk.CTkSlider(self.ctrl_frame, from_=0.0, to=1.0, number_of_steps=100)
        self.w_resource.set(0.34)
        self.w_resource.pack(pady=6, padx=12)
        self.w_resource_label = ctk.CTkLabel(self.ctrl_frame, text="W_resource: 0.34")
        self.w_resource_label.pack()

        # Update labels when sliders move
        self.w_delay.configure(command=self._update_weight_labels)
        self.w_reliability.configure(command=self._update_weight_labels)
        self.w_resource.configure(command=self._update_weight_labels)

        # Calculate button
        self.calc_btn = ctk.CTkButton(self.ctrl_frame, text="Hitung", command=self.on_calculate)
        self.calc_btn.pack(pady=(12, 8))

        # Results panel
        self.results_frame = ctk.CTkFrame(self.ctrl_frame)
        self.results_frame.pack(pady=(8, 12), fill='both', expand=False)

        self.res_delay = ctk.CTkLabel(self.results_frame, text="Total Delay: -")
        self.res_delay.pack(pady=6)
        self.res_reliability = ctk.CTkLabel(self.results_frame, text="Total Reliability: -")
        self.res_reliability.pack(pady=6)
        self.res_resource = ctk.CTkLabel(self.results_frame, text="Resource Cost: -")
        self.res_resource.pack(pady=6)

        # Initial draw
        self.draw_graph()

    def _update_weight_labels(self, _=None):
        d = self.w_delay.get()
        r = self.w_reliability.get()
        s = self.w_resource.get()
        total = d + r + s
        if total == 0:
            nd, nr, ns = 1/3, 1/3, 1/3
        else:
            nd, nr, ns = d/total, r/total, s/total
        self.w_delay_label.configure(text=f"W_delay: {nd:.3f}")
        self.w_reliability_label.configure(text=f"W_reliability: {nr:.3f}")
        self.w_resource_label.configure(text=f"W_resource: {ns:.3f}")

    def draw_graph(self, highlight_path=None):
        self.ax.clear()
        node_colors = [self.graph.nodes[n].get('NodeReliability', 1.0) for n in self.graph.nodes()]
        nodes = list(self.graph.nodes())
        cmap = plt.cm.viridis
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=nodes, node_size=20,
                               node_color=node_colors, cmap=cmap, ax=self.ax)
        nx.draw_networkx_edges(self.graph, self.pos, alpha=0.25, ax=self.ax)

        if highlight_path and len(highlight_path) > 1:
            edge_list = list(zip(highlight_path[:-1], highlight_path[1:]))
            nx.draw_networkx_nodes(self.graph, self.pos, nodelist=highlight_path, node_size=60, node_color='red', ax=self.ax)
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=edge_list, width=2.5, edge_color='red', ax=self.ax)

        self.ax.set_axis_off()
        self.canvas.draw()

    def on_calculate(self):
        # read inputs
        try:
            source = int(self.s_var.get())
            dest = int(self.d_var.get())
        except Exception:
            ctk.CTkMessagebox = None
            return

        # normalize weights
        d = self.w_delay.get()
        r = self.w_reliability.get()
        s = self.w_resource.get()
        total = d + r + s
        if total == 0:
            w_delay, w_reliability, w_resource = 1/3, 1/3, 1/3
        else:
            w_delay, w_reliability, w_resource = d/total, r/total, s/total

        # Placeholder algorithm: use shortest path (by hop count) as a quick demo
        try:
            path = nx.shortest_path(self.graph, source=source, target=dest)
        except Exception:
            path = []

        # simulate iterative improvement so UI shows real-time updates
        steps = 40
        last_metrics = (0.0, 0.0, 0.0)
        for i in range(steps):
            # simulate small changes to metrics
            td, tr, rc = compute_path_metrics(self.graph, path)
            # apply weights to compute a synthetic score (not used further here)
            score = w_reliability * tr - w_delay * td - w_resource * rc
            # update labels progressively
            self.res_delay.configure(text=f"Total Delay: {td:.3f} ms")
            self.res_reliability.configure(text=f"Total Reliability: {tr:.6f}")
            self.res_resource.configure(text=f"Resource Cost: {rc:.6f}")
            self.update_idletasks()
            self.after(25)

        # final draw with highlighted path
        self.draw_graph(highlight_path=path)


if __name__ == '__main__':
    G = create_erdos_renyi_connected()
    app = RoutingGUI(graph=G)
    app.mainloop()
