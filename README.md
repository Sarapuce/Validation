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
If not precised, file name column label MUST start with 'file'. Else, you can precise it by giving the list of file labels with the argument x_name

| |file_0         |file_1         |y_true|class_1|class_2|class_3|
|-|---------------|---------------|------|-------|-------|-------|
|0|class_1_x20.png|class_1_x10.png|1     |0.84   |0.10   |0.06   |
|1|class_2_x20.png|class_2_x10.png|2     |0.75   |0.05   |0.20   |
|2|class_3_x20.png|class_3_x10.png|3     |0.10   |0.10   |0.80   |

Or
```
from validation import Validation
validation = Validation('path_df/file.df', 'path_image_folder', x_name = ['file_0', 'file_1'])
validation.report('path_save/name.pdf', 'My personnel report on Mnist')
```
| WARNING: Classes order is important |
| --- |
