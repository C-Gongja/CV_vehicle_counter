# CV_vehicle_counter

## Description
Count the number of vehicles from video or an image.

## Motivation
Vehicle counting can be used for analyzing traffic patterns, which can be applied to studying traffic management, transportation metrics, and the environmental impact of vehicles. It can also be used to correctly identify other cars in autonomous driving cars. 

## Data source
  * Title: An autonomous driving dataset <br>
  * [Waymo Open Dataset](https://www.waymo.com/open) <br>
  * Year: 2019

## Methods
1. Frame differencing <br>
   - Compare consecutive frames in a video to detect changes in pixel values. This is used for motion detection. 
2. Image thresholding
   - Convert each frame into a binary image. Each pixel is classified as either foreground (white) or background (black) based on their intensity and color values.
3. Contour finding
   - Identifies and extracts the boundaries of objects of interest in an image. These boundaries are represented as continuous curves or outlines that trace the edges of objects in an image. This is used for object recognition and shape analysis.
4. Image dilation
   - Expand the boundaries of objects in the binary images. This thickens regions of interest, making it easier to detect and analyze the vehicles.
5. Vehicle counting
   - Count the number of vehicles based on certain criteria (size, shape, motion patterns, etc.)
