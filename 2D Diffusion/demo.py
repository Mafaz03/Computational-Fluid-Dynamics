import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

# Example function
def plot_from_inputs(t1, t2, t3, t4, t5):
    # Convert to numbers if possible
    try:
        values = [float(t1), float(t2), float(t3), float(t4), float(t5)]
    except:
        values = [0, 0, 0, 0, 0]

    # Create a plot
    fig, ax = plt.subplots()
    ax.plot(values, marker='o')
    ax.set_title("Plot from Inputs")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")

    return fig

# Define Gradio Interface
demo = gr.Interface(
    fn=plot_from_inputs,
    inputs=[
        gr.Textbox(label="Input 1"),
        gr.Textbox(label="Input 2"),
        gr.Textbox(label="Input 3"),
        gr.Textbox(label="Input 4"),
        gr.Textbox(label="Input 5"),
    ],
    outputs=gr.Plot(label="Generated Plot"),
)

if __name__ == "__main__":
    demo.launch()