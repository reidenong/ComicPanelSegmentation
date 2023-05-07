# Comic Panel Segmentation Tool
A tool using classical computer vision for segmentation of digital images composing of multiple panels. Especially effective on Comic panels, Collages and Memes. 

<br />

Segmentation on complex image collage formats helps to improves performance of downstream tasks such as OCR, Object detection and other computer vision tools in an image processing pipeline.

<br />

![image](https://user-images.githubusercontent.com/65756407/236692808-b3cef540-f75a-4be9-9ec4-5d48fe5f2132.png)


<br />

## Examples of PST with other internet images

![image](https://user-images.githubusercontent.com/65756407/236692909-adcd233c-8c18-4566-b652-56b0e331375c.png)



<br />

### About the comic Panel Segmentation Tool
While many image segmentation tools these days take a [deep learning approach](https://github.com/facebookresearch/segment-anything), PST chooses to rely on classical computer vision. This is possible due to the relatively small problem space of comics and digitally created image formats; coupled with exploiting some recurring tendencies such as pixel perfect panel delineations, this allows for accurate segmentation to be done with some clever heuristics.

<br />

Key benefits of PST over its deep learning counterparts include faster processing speeds as well as being much more lightweight. PST struggles with non computer generated graphics which are hand-drawn, images in the wild or unconventional internet image formats with no clear delineations between panels. However it is still very effective for images found on the internet such as screenshots, collages, memes, comics etc., with most other cases probably requiring a deep learning approach due to the broad problem space.

<br />

### Brief Methodology
1. Canny Edge Detection is used to detect the presence of all edges in the image
2. A Probabilistic Hough Transformation is used to detect all straight lines, with filters of length and angle classifiers to eliminate all noisy lines.
3. Resultant lines are then extrapolated and merged on the pixel level with other lines, with many lines congregating along key delimiting lines between panels, which are termed as segment lines.
4. Segmentation Lines are then evaluated with other segmentation lines on the pixel level to determine likelihood of a panel edge existing there. Most of the irrelevant noise of naturally occuring straight lines in an image ar eliminated in this step.
4. Images are sliced, and positional information of each panel is retained in bounding box notation and optionally returned in JSON format. 

![image](https://user-images.githubusercontent.com/65756407/236691577-4f53f630-e7c8-46dd-82e5-f20d277ba0bc.png)



<br />

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
       [--bboxjson]
       [--pdf]
       
optional arguments:
    --help
    show this help message and exit
    
    --filepath your/filepath/here
    directories where images are received as input, default is set as 'Data/'
    
    --outputfilepath your/filepath/here
    directories where images are saved, default is set as 'Output/'
    
    --nosaveimage
    choose not to save images
    
    --bbox json
    choose to save image segmentation bounding boxes, with the same filename in JSON
    
    --pdf
    generates a pdf useful for troubleshooting
```

<br />

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

