# PanelSegmentationTool
A tool using classical computer vision for segmentation of digital images composing of multiple panels. Especially effective on Comic panels, Collages and Memes.

Brief Methodology
- Canny Edge Detection is used to detect the presence of edges
- A Hough Transformation is used to detect and filter out lines likely to be Segmentation Lines
- Segmentation Lines are then evaluated with other segmentation lines to determine likelihood of a panel edge existing there, whereupon most noise is filtered out
- Images are sliced
