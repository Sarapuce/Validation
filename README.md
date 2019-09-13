# Validation
Little python script to analyse output of a net

### Usage
```
from validation import Validation
validation = Validation('path_df/file.df', 'path_image_folder')
validation.report('path_save/name.pdf', 'My personnel report on Mnist')
```

### DataFrame structure

| |file       |y_true|class_1|class_2|class_3|
|-|-----------|------|-------|-------|-------|
|0|class_1.png|1     |0.84   |0.10   |0.06   |
|1|class_2.png|2     |0.75   |0.05   |0.20   |
|2|class_3.png|3     |0.10   |0.10   |0.80   |

### Support multi-image classification

| |file_0         |file_1         |y_true|class_1|class_2|class_3|
|-|---------------|---------------|------|-------|-------|-------|
|0|class_1_x20.png|class_1_x10.png|1     |0.84   |0.10   |0.06   |
|1|class_2_x20.png|class_2_x10.png|2     |0.75   |0.05   |0.20   |
|2|class_3_x20.png|class_3_x10.png|3     |0.10   |0.10   |0.80   |
