import data_pipeline

ds_train, dataset_info = data_pipeline.get_data(data_dir='datasets/Sketches',
                                                img_size=256,
                                                img_channels=3,
                                                num_classes=0,
                                                num_devices=1,
                                                batch_size=5)
