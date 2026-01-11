import gradio as gr

def passthrough(frame):
    return frame

with gr.Blocks() as demo:
    gr.HTML("<h3>👆 Click 'Record' to start live face recognition</h3>")
    cam = gr.Image(
        sources=["webcam"],
        streaming=True,
        label="Live Camera"
    )
    out = gr.Image()

    cam.stream(passthrough, inputs=cam, outputs=out)

demo.launch()
