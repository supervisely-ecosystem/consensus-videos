<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/consensus/assets/61844772/8132fd69-23a8-4407-8316-48d21430717e"/> 

# Labeling Consensus for Videos
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#What-Report-metrics-mean">What Report metrics mean</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/consensus)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/consensus)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/consensus.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/consensus.png)](https://supervise.ly)

</div>

# Overview

This application allows you to compare annotations made by different users, generate a report with metrics and perform actions with images based on the comparison results.

# How To Run

**Step 1:** Run the application from the ecosystem.

**Step 2:** Wait until the app starts.

Once the app is started, a new task appear in workspace tasks. Wait for the message `Application is started ...` and then press `Open` button.

**Step 3:** Open the app.

**Step 4:** Select projects, datasets and annotators to compare.

You can select multiple annotators and datasets. You can see added annotations in the table to the right. To remove an annotation from comparison, select it in the table and press the `Remove` button.

<p align="center"><img src="https://github.com/supervisely-ecosystem/consensus/assets/61844772/20b9def0-5f1b-4871-917d-bbcea5c9e40e" /></p>

**Step 5:** Run the comparison.

Press the `Calculate consensus` button to start the comparison. The comparison may take a long time, depending on the number of images and annotations. After the calculation is finished, you will see the results in the comparison matrix.

<p align="center"><img src="https://github.com/supervisely-ecosystem/consensus/assets/61844772/0020ad34-a1d6-4414-acb8-50af3caa6a71" /></p>

**Step 6:** View the detailed report.

Click on a cell of the comparison matrix with a score to see a detailed report for the given pair of annotations.

<p align="center"><img src="https://github.com/supervisely-ecosystem/consensus/assets/61844772/c15e131b-9154-45f9-a11c-5a5381497d51" /></p>

# What Report metrics mean

### Table **Objects counts per class**
- **GT Objects** - Total number of objects of the given class on all images in the benchmark dataset
- **Labeled Objects** - Total number of objects of the given class on all images in the exam dataset
- **Recall(Matched objects)** - Number of correctly labeled objects of the given class divided by the total number of objects of the given class on all images in the benchmark dataset.
- **Precision** - Number of correctly labeled objects of the given class divided by the total number of objects of the given class on all images in the exam dataset
- **F-measure** - Harmonic mean of Recall and Precision
- **Objects Score (average F-measure)** - Average of F-measures of all classes

### Table **Geometry quality**
- **Pixel Accuracy** - Number of correctly labeled or unlabeled pixels on all images divided by the total number of pixels
- **IOU** - Intersection over union. Sum of intersection areas of all objects on all images divided by the sum of union areas of all objects on all images for the given class
- **Geometry Score (average IoU)** - Average of IOUs of all classes

### Table **Tags**
- **GT Tags** - Total number of tags of all objects on all images in the benchmark dataset
- **Labeled Tags** - Total number of tags of all objects on all images in the exam dataset
- **Precision** - Number of correctly labeled tags divided by the total number of tags of all objects on all images in the exam dataset
- **Recall** - Number of correctly labeled tags divided by the total number of tags of all objects on all images in the benchmark dataset
- **F-measure** - Harmonic mean of Recall and Precision
- **Tags Score** - Average of F-measures of all tags

### Table **Report per image**
- **Objects Score** - F-measure of objects of any class on the given image
- **Objects Missing** - Number of objects of any class on the given image that are not labeled
- **Objects False Postitve** - Number of objects of any class on the given image that are labeled but not present in the benchmark dataset
- **Tags Score** - F-measure of tags of any object on the given image
- **Tags Missing** - Number of tags of any object on the given image that are not labeled
- **Tags False Positive** - Number of tags of any object on the given image that are labeled but not present in the benchmark dataset
- **Geometry Score** - Intersection over union for all objects of any class on the given image
- **Overall Score** - Average of Objects Score, Tags Score and Geometry Score
