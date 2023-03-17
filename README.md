# PanelSegmentationTool (PST)
A tool using classical computer vision for segmentation of digital images composing of multiple panels. Especially effective on Comic panels, Collages and Memes. Segmentation on complex image collage formats helps to improves performance of downstream tasks such as OCR, Object detection and other computer vision tools in an image processing pipieline.

### Installing Dependencies
```$ pip install -r requirements.txt```

### Usage
```
$ python3 PST.py

usage: PST.py
       [-h]
       [--filepath FILEPATH]
       [--outputfilepath OUTPUTFILEPATH]
       [--nosaveimage]
       [--pdf]
       
optional arguments:
    --help
    show this help message and exit
    
    --filepath
    directories where images are received as input, default is set as 'Data/'
    
    --outputfilepath
    directories where images are saved, default is set as 'Output/'
    
    --nosaveimage
    choose not to save images
    
    --pdf
    generates a pdf useful for troubleshooting
```

### Directory Structure
```
├── Main
│   ├── PST.py
│   ├── Data
│   │   ├── image0.jpg
│   │   └── image1.jpg
.
.
.
│   ├── Output
│   │   ├── image0
│   │   │   ├── 0.jpg
│   │   │   └── 1.jpg
│   │   ├── image1
│   │   │   ├── 0.jpg
│   │   │   └── ...
...
```
### Brief Methodology
- Canny Edge Detection is used to detect the presence of edges
- A Hough Transformation is used to detect and filter out lines likely to be Segmentation Lines
- Segmentation Lines are then evaluated with other segmentation lines to determine likelihood of a panel edge existing there, whereupon most noise is filtered out
- Images are sliced

![Capture](https://user-images.githubusercontent.com/65756407/225867600-f44cb719-b8a4-41c3-92fb-876e6517675f.JPG)
