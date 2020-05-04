# Project Write-Up

## Explaining Custom Layers

In the model used, the layers unsupported by the intel hardware are supported with the extension `/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so`.

Some of the potential reasons for handling custom layers are time when some layers are not supported by openvino inference engine. No custom layers were handled in this project.

## Comparing Model Performance

My method to compare models before and after conversion to Intermediate Representations
was to load and run the detection in the original framework and then on the inference engine. I used `benchmark.py` script to achieve the time taken for the two of them.

| Type                           | ssd_mobilenet_v1_coco | ssd_mobilenet_v2_coco | ssd_inception_v2_coco | Pre-trained model    |
| ------------------------------ | --------------------- | --------------------- | --------------------- | -------------------- |
| ProtoBuf size                  | 28M                   | 69M                   | 102M                  | NA                   | 
| Bin size                       | 26M	                 | 65M                   | 96M                   | 5.1M                 |
| Xml size                       | 80K	                 | 112K                  | 148K                  | 88K                  |
| Pre: Load model + Infer Time   | 2 minute 55 seconds   | 3 minutes 55 seconds  | 5 minutes 47 seconds  | NA                   |
| Post: Load model + Infer Time  | 1 minute 39 seconds   | 2 minutes 5 seconds   | 2 minutes 4 seconds   | 1 minutes 47 seconds |
| Accuracy                       | Low                   | Low                   | Low                   | High                 |
| Network need & cost            | High                  | High                  | High                  | Low                  |

Note: 
 * Pre-trained model chosen was `pedestrian-detection-adas-0002`.
 * `probability-threshold` was set to `0.6` for above measurements.
 * Network need and the cost of using cloud servcie as opposed to using edge is directly proportional to the size of the model.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are departmental stores, buses and train & airport platforms to track number of people inside.

Each of these use cases would be useful because it would help security and other management officials to control traffic.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows

* `Lighting` - It depends on the pre-processing techniques used before the image is input to the inference engine. If proper filters are applied, lighting effects could be handled to a good extend to have its impact on the outcome.
* `Accuracy` - It is crucial since we are afford to loose only a few frames in between the proper detection. Otherwise it might be difficult to track the person and hence it might have a big impact on the resulting count and duration.
* `Image Size` - In general, a model used might be trained for specific input image and target size. So the having a drastically different input image size and scaling it to the required size might have a negative impact on the performance.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Models: [ssd_mobilenet_v1_coco, ssd_mobilenet_v2_coco and ssd_inception_v2_coco]
  - [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
  - `Steps`: I converted the models to an Intermediate Representation with the following commands  
    * `ssd_mobilenet_v1_coco`
    ```
    $ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    $ tar -xvpf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    $ cd ssd_mobilenet_v1_coco_2018_01_28/
    $ python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json
    ```
    * `ssd_mobilenet_v2_coco`
    ```
    $ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    $ tar -xvpf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    $ cd ssd_mobilenet_v2_coco_2018_03_29/
    $ python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```
    * `ssd_inception_v2_coco`
    ```
    $ wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.tar.gz
    $ tar -xvpf ssd_inception_v2_coco_2018_01_28.tar.tar.gz
    $ cd ssd_inception_v2_coco_2018_01_28.tar/
    $ python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```
  - `Reason of Failure`: The models were insufficient for the app because the detection accuracies were low. So they were not giving the valid boxes for frames in between some times and hence it was not possible to keep track of the count and duration.
  - `Trials`:
    - I tried to improve the app by applying some logics like checking threshold value for duration before the state change to avoid false trigger due to dropped frames and then again by tracking the position of the people to eliminate false triggers. But still it didn't help to get the desired result.
    - I tried to adjust the confidence threshold value which also didn't result in any improvement in the accuracy.

- `Final Model`: After the above trials, I decided move to the pre-trained IRs. I chose, `pedestrian-detection-adas-0002`(32-bit version) for my app.
  - `Steps`: I used the following commands to download the model
    ```
    $ cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
    $ ./downloader.py --name pedestrian-detection-adas-0002 -o /home/workspace
    ```
  - `Testing`: When I tested, it met all the performance criteria mentioned in the rubric, It was lighter in size and the output inference time was lesser. And most of all the accuracy was almost perfect. Which was way better compared to the other previous models which were considered.
