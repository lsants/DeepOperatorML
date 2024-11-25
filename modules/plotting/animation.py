import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import animation

def animate_wave(g_u, g_u_pred=None, interval=20, fps=30, save_name=None):

    fig, ax = plt.subplots()
    if g_u_pred is not None:
        assert g_u.shape == g_u_pred.shape
        lims = (np.min([g_u.min(), g_u_pred.min()]), np.max([g_u.max(), g_u_pred.max()]))
    else:
        lims = (g_u.min(), g_u.max())

    # Plot the first frame
    surface1 = ax.imshow(g_u[0], animated=True, vmin=lims[0], vmax=lims[1])
    if g_u_pred is not None:
        surface2 = ax.imshow(g_u_pred[0], animated=True, alpha=0.5, vmin=lims[0], vmax=lims[1])
        args = (g_u, g_u_pred,)
    else:
        args = (g_u,)

    def _animate(i, u, u_pred=None):
        surface1.set_data(u[i])
        if u_pred is not None:
            surface2.set_data(u_pred[i])
            return [surface1, surface2]
        else:
            return [surface1]
        
    anim = animation.FuncAnimation(
        fig,
        _animate,
        fargs=args,
        interval=interval,
        blit=True,
        frames=range(g_u.shape[0])
    )

    # Display the video
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)

    # Save the video if a filename is provided
    if save_name is not None:
        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(f"{save_name}.mp4", writer=writervideo)

    plt.close()
