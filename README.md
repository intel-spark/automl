# AutoML library

It consists of an automated machine learning libary based on Ray, and a specially designed and auto-tuned package for time sequence prediction.

## Interfaces of AutoML library

- ```FeatureTransformer```

- ```Model```

- ```Pipeline```

- ```SearchEngine```

## Example Usage for Time Sequence Prediction

Current implementation only supports univariant prediction, which means features can be multivariant, while target value should only be one on each data point of the sequence.  

```automl.regression.TimeSequencePredictor.fit```

```automl.regression.TimeSequencePredictor.evaluate ```

```automl.regression.TimeSequencePredictor.predict```
