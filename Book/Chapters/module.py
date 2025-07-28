import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

class LinearSystemSolverNotebook:
    def __init__(self):
        self.default_eqs = [
            [20, 50, 700],
            [1, 1, 20],
            [50, 20, 700]
        ]
        self.equation_widgets = []
        self.third_eq_shown = False

        # UI elements
        self.output_area = widgets.Output()
        self.plot_area = widgets.Output()
        self.toggle_button = widgets.ToggleButton(
            value=False,
            description='[+]',
            tooltip='Toggle third equation',
            layout=widgets.Layout(width='50px')
        )
        self.toggle_button.observe(self.toggle_third_equation, names='value')

        self.create_equation_rows(2)

        # Display layout
        self.container = widgets.VBox(
            self.equation_widgets + [self.toggle_button, self.output_area, self.plot_area]
        )
        display(self.container)

        self.update_plot()

    def create_equation_rows(self, count):
        for i in range(count):
            self.add_equation_row(i)

    def add_equation_row(self, idx):
        default = self.default_eqs[idx]
        a = widgets.Text(value=str(default[0]), layout=widgets.Layout(width='60px'))
        c = widgets.Text(value=str(default[1]), layout=widgets.Layout(width='60px'))
        b = widgets.Text(value=str(default[2]), layout=widgets.Layout(width='60px'))
        for box in (a, c, b):
            box.observe(lambda change: self.update_plot(), names='value')

        row = widgets.HBox([
            a, widgets.Label("c +"),
            c, widgets.Label("t ="),
            b
        ])
        row._entries = [a, c, b]
        self.equation_widgets.append(row)

    def toggle_third_equation(self, change):
        if change['new']:
            self.add_equation_row(2)
            self.container.children = self.equation_widgets + [self.toggle_button, self.output_area, self.plot_area]
            self.toggle_button.description = '[-]'
            self.third_eq_shown = True
        else:
            self.equation_widgets.pop()
            self.container.children = self.equation_widgets + [self.toggle_button, self.output_area, self.plot_area]
            self.toggle_button.description = '[+]'
            self.third_eq_shown = False
        self.update_plot()

    def parse_equations(self):
        A, b = [], []
        for row in self.equation_widgets:
            a, c, b_ = row._entries
            try:
                a_val = float(a.value) if a.value.strip() else 1.0
                c_val = float(c.value) if c.value.strip() else 1.0
                b_val = float(b_.value) if b_.value.strip() else 1.0
                A.append([a_val, c_val])
                b.append(b_val)
            except ValueError:
                continue
        return np.array(A), np.array(b)

    def update_plot(self):
        A, b = self.parse_equations()

        with self.output_area:
            clear_output()
            if len(A) < 2:
                print("Enter at least 2 valid equations.")
                return

            print("     A         x    =   b")
            for i in range(len(A)):
                a_str = f"[{A[i,0]:>4.0f} {A[i,1]:>4.0f}]"
                x_str = "[c]" if i == 0 else "[t]" if i == 1 else "   "
                b_str = f"[{b[i]:>5.0f}]"
                print(f"{a_str}   {x_str}   {b_str}")

            # Initialize flags
            sol = None
            show_dot = False
            msg = ""
            title_color = 'black'

            try:
                rank_A = np.linalg.matrix_rank(A)
                Ab = np.hstack([A, b.reshape(-1, 1)])
                rank_Ab = np.linalg.matrix_rank(Ab)

                if rank_Ab != rank_A:
                    msg = "System is inconsistent: No solution."
                    title_color = 'red'
                elif rank_A < A.shape[1]:
                    msg = "System is dependent: Infinite solutions."
                    title_color = 'blue'
                else:
                    if A.shape[0] == A.shape[1]:
                        sol = np.linalg.solve(A, b)
                        msg = f"Unique solution: c = {sol[0]:.2f}"
                        show_dot = True
                        title_color = 'green'
                    else:
                        sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                        msg = f"Least squares solution: c = {sol[0]:.2f}"
                        show_dot = True
                        title_color = 'gray'
            except Exception as e:
                msg = f"Error solving system: {e}"

            print("\n" + msg)

            try:
                cond = np.linalg.cond(A)
                print(f"Condition number: {cond:.2e}")
            except:
                print("Condition number: N/A")

        # Plotting
        with self.plot_area:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))

            # Center on solution if available
            if sol is not None:
                x_center, y_center = sol
                x_min, x_max = x_center - 20, x_center + 20
                y_min, y_max = y_center - 20, y_center + 20
            else:
                x_min, x_max = -40, 40
                y_min, y_max = -40, 40

            x_vals = np.linspace(x_min, x_max, 1000)

            for i in range(len(A)):
                a, c = A[i]
                rhs = b[i]
                label = f"{a:.0f}c + {c:.0f}t = {rhs:.0f}"

                if c != 0:
                    y_vals = (rhs - a * x_vals) / c
                    ax.plot(x_vals, y_vals, label=label)
                elif a != 0:
                    ax.axvline(x=rhs / a, label=label)
                else:
                    ax.axhline(y=rhs, label=label)

            if sol is not None and show_dot:
                ax.plot(*sol, 'ro', label="Solution")

            ax.set_title(msg, color=title_color)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', lw=0.5)
            ax.axvline(0, color='black', lw=0.5)
            ax.set_xlabel("c")
            ax.set_ylabel("t")
            ax.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

LinearSystemSolverNotebook()
