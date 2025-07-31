import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

class LinearSystemSolverPlotly:
    def __init__(self):
        # Default equations: [a, c, b]
        self.default_eqs = [
            [20, 50, 700],
            [ 1,  1,  20],
            [50, 20, 700]
        ]
        self.equation_widgets = []

        # Output area & toggle button (start with 3 equations)
        self.output_area = widgets.Output()
        self.toggle_button = widgets.ToggleButton(
            value=True,
            description='[-]',
            tooltip='Toggle third equation',
            layout=widgets.Layout(width='50px')
        )
        self.toggle_button.observe(self._on_toggle, names='value')

        # Plotly figure widget
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            width=600, height=600,
            margin=dict(l=40, r=40, b=40, t=40),
            xaxis=dict(title='c'),
            yaxis=dict(title='t'),
            showlegend=True
        )

        # Build 3 equation rows and track the third
        for idx in range(3):
            row = self._add_eq_row(idx)
            if idx == 2:
                self.third_row = row

        # Start with third row visible
        self.third_row.layout.display = None

        # Layout & display
        self.container = widgets.VBox(
            self.equation_widgets +
            [self.toggle_button, self.output_area, self.fig]
        )
        display(self.container)

        # Initial plot
        self.update_plot()

    def _add_eq_row(self, idx):
        a = widgets.FloatText(value=self.default_eqs[idx][0], layout=widgets.Layout(width='60px'))
        c = widgets.FloatText(value=self.default_eqs[idx][1], layout=widgets.Layout(width='60px'))
        b = widgets.FloatText(value=self.default_eqs[idx][2], layout=widgets.Layout(width='60px'))
        for w in (a, c, b):
            w.observe(lambda change: self.update_plot(), names='value')
        row = widgets.HBox([a, widgets.Label("c +"), c, widgets.Label("t ="), b])
        row._entries = (a, c, b)
        self.equation_widgets.append(row)
        return row

    def _on_toggle(self, change):
        if change['new']:
            # Show third row
            self.third_row.layout.display = None
            self.toggle_button.description = '[-]'
        else:
            # Hide third row
            self.third_row.layout.display = 'none'
            self.toggle_button.description = '[+]'
        self.update_plot()

    def parse_equations(self):
        A, b = [], []
        for row in self.equation_widgets:
            # Skip hidden third row
            if getattr(row.layout, 'display', None) == 'none':
                continue
            a, c, b_ = row._entries
            A.append([a.value, c.value])
            b.append(b_.value)
        return np.array(A), np.array(b)

    def update_plot(self):
        A, b = self.parse_equations()

        # Clear previous traces and title
        self.fig.data = []
        self.fig.layout.title = ""
        self.fig.layout.showlegend = True

        with self.output_area:
            clear_output()
            if A.shape[0] < 2:
                print("Enter at least 2 valid equations.")
                return

            # Print matrix form
            print("     A         x    =   b")
            for i in range(A.shape[0]):
                a_str = f"[{A[i,0]:>4.0f} {A[i,1]:>4.0f}]"
                x_str = "[c]" if i==0 else "[t]" if i==1 else "   "
                b_str = f"[{b[i]:>5.0f}]"
                print(f"{a_str}   {x_str}   {b_str}")

            rank_A  = np.linalg.matrix_rank(A)
            rank_Ab = np.linalg.matrix_rank(np.hstack([A, b.reshape(-1,1)]))

            sol = None
            show_dot = False

            if A.shape[0] == 3:
                # Always least-squares for 3 eqns
                sol, residuals, *_ = np.linalg.lstsq(A, b, rcond=None)
                rss = residuals[0] if residuals.size else 0.0
                if rank_Ab > rank_A:
                    msg, color = f"Inconsistent: LS fit (RSS={rss:.1f})", 'red'
                elif rank_A < A.shape[1]:
                    msg, color = f"Dependent: LS fit (RSS={rss:.1f})", 'blue'
                elif np.isclose(rss, 0):
                    msg, color = f"Exact intersection: c={sol[0]:.2f}, t={sol[1]:.2f}", 'green'
                else:
                    msg, color = f"Least-squares: c={sol[0]:.2f}, t={sol[1]:.2f} (RSS={rss:.1f})", 'gray'
                show_dot = True
            else:
                # Two eqns: decide exact or no/infinite solutions
                if rank_Ab > rank_A:
                    msg, color, show_dot = "Inconsistent: no solution.", 'red', False
                elif rank_A < A.shape[1]:
                    msg, color, show_dot = "Dependent: infinite solutions.", 'blue', False
                else:
                    sol = np.linalg.solve(A, b)
                    msg, color, show_dot = (f"Solution: c={sol[0]:.2f}, t={sol[1]:.2f}", 'green', True)

            print("\n" + msg)
            try:
                cond = np.linalg.cond(A)
                print(f"Condition number: {cond:.2e}")
            except:
                print("Condition number: N/A")

        # Determine plot range
        if sol is not None:
            cx, cy = sol
            x_min, x_max = cx-20, cx+20
            y_min, y_max = cy-20, cy+20
        else:
            x_min, x_max, y_min, y_max = -40, 40, -40, 40

        x_vals = np.linspace(x_min, x_max, 500)
        # Plot each visible line
        for i in range(A.shape[0]):
            a, c = A[i]
            rhs = b[i]
            label = f"{a}c + {c}t = {rhs}"
            if c != 0:
                yv = (rhs - a*x_vals)/c
                self.fig.add_trace(go.Scatter(x=x_vals, y=yv, mode='lines', name=label))
            elif a != 0:
                xc = np.full_like(x_vals, rhs/a)
                self.fig.add_trace(go.Scatter(x=xc, y=x_vals, mode='lines', name=label))
            else:
                yc = np.full_like(x_vals, rhs)
                self.fig.add_trace(go.Scatter(x=x_vals, y=yc, mode='lines', name=label))

        # Plot solution/fit dot
        if show_dot:
            self.fig.add_trace(go.Scatter(
                x=[sol[0]], y=[sol[1]],
                mode='markers', name='Solution',
                marker=dict(color='red', size=10)
            ))

        # Final layout
        self.fig.update_layout(
            title=dict(text=msg, font=dict(color=color)),
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )

LinearSystemSolverPlotly()
