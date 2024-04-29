using FFMPEG
imagesdirectory = "figures"
# imagesdirectory = "."
framerate = 30
gifname = "/fig_map_animation.gif"
FFMPEG.ffmpeg_exe(`-framerate $(framerate) -f image2 -i $(imagesdirectory)/fig_map_%3d.png -y $(gifname)`)
