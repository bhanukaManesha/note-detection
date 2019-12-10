## Cash Labeling Script

Run the following command in the terminal

```
python cashLabeling.py "dataset/videos/greenscreen_50.MOV" "dataset/image/RM50/" True 0 
```

#### The arguments for the script
```
python cashLabeling.py <video_path> <output_path> <collect_data> <verbose>
```
- video_path : the path to the video file
- output_path : the path the output file
- collect_data : flag to indicate whether to collect the data (`True` or `False`)
- verbose : 
  - `0` - Only progress bar
  - `1` - File name and bounding box
  - `2` - Video output
