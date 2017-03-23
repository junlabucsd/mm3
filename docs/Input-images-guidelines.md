# Input images guidelines

For mm3 to properly process image data there are some general guidelines you should try to follow when collecting and pre-processing your data. Problems can manifest themselves when trying to find channels as well as extracting metadata. Here is a collection of guidelines and assumptions of the program:

### Images are individual time points and separated by FOV.

mm3 expects that each image is a separate time point. It cannot currently negotiate a single stacked TIFF. FOVs should also be separated.

### Images with multiple planes (colors) are stacked.

If the experiment has multiple planes, for example a phase contrast and fluorescent image, those planes should be in the same stack. When slicing out channels, mm3 will slice through all planes and then save them separately. 

### Channels are mostly vertical.

This is important in order to find the channels, which is done with a 1D wave convolution.
It is also important when slicing out the individual channels, which assumes vertical channels with some additional padding.

### Channels have at least 20 pixels from their ends to the top and bottom edge of the image.

This is not a hard and fast rule, but insures that when slicing out channels with some padding, no area outside the image is ever referenced. Channels closer to the left and right side of the image by less than half the distance between two channels are ignored when finding channels.

### Channel ends are in the top and bottom third of the image, regardless of orientation.

When finding the closed and open end of the channels, the software only looks in the upper and lower third. This is also important for automatically determining the orientation of the images (channel open-end is down or up). If the channels are not roughly centered vertically (i.e. there is a lot of extra image either above or below the channels), this could mess up automatic orientation and finding the channel ends.

### There are no artifacts in the image above and below the channels.

This also can create a problem when finding channels. For example, the numbers on the mother machine can confuse the script, which uses a vertical image profile to find the open and closed end of the channels.
