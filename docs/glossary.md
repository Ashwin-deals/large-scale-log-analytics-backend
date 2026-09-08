# MorphGuard — Glossary of Terms

---

This glossary defines key terms, acronyms, and concepts used throughout the MorphGuard project documentation and codebase.

---

## A

### Accuracy
The proportion of all predictions (both Normal and Anomaly) that are correct. In the context of MorphGuard, accuracy is a misleading metric due to the severe class imbalance (~97.1% Normal blocks), since a trivial classifier predicting "Normal" for everything achieves ~97.1% accuracy while detecting zero anomalies.

### Anomaly
In the context of MorphGuard, an HDFS block whose lifecycle exhibits behavior patterns that deviate significantly from normal operations. Anomalies may indicate hardware failures, software bugs, security incidents, or configuration errors. In the HDFS dataset, approximately 2.9% of blocks are labeled as anomalous.

### Anomaly Score
A numerical value produced by the Isolation Forest model indicating how anomalous a data point is. Higher scores (computed as the negated decision function) indicate more anomalous behavior. The score is a continuous value, while the predicted label (Normal/Anomaly) is derived from a threshold determined by the contamination parameter.

### Atlas (MongoDB Atlas)
MongoDB's cloud-hosted database service. MorphGuard supports Atlas connections via the certifi TLS bundle, enabling deployment without local MongoDB installation.

---

## B

### BaseParser
The abstract base class in `parser/base_parser.py` that defines the interface all log parsers must implement. It requires a `parse()` method that accepts a filepath and returns a pandas DataFrame.

### bcrypt
A password hashing algorithm used by MorphGuard to securely store user passwords. bcrypt automatically generates a salt for each password and is resistant to rainbow table attacks and brute force due to its configurable work factor.

### Block ID
A unique identifier assigned to each data block in HDFS (e.g., `blk_38865049064139660`). In MorphGuard, the block_id serves as the primary aggregation key for feature engineering and the unit at which anomaly labels are assigned.

---

## C

### Chromosome
In the Genetic Algorithm, a chromosome is a fixed-length vector of genes that encodes a complete candidate solution. In MorphGuard, a chromosome contains N binary feature selection genes followed by 4 hyperparameter genes (n_estimators, max_samples, max_features, contamination).

### Class Imbalance
A dataset characteristic where one class is significantly more frequent than another. The HDFS dataset has a 33:1 ratio of Normal to Anomaly blocks (~97.1% vs ~2.9%), making metrics like accuracy unreliable and necessitating the use of F1 score.

### Confusion Matrix
A table showing the counts of true positives, false positives, true negatives, and false negatives. In MorphGuard, the rows represent true labels and the columns represent predicted labels.

### Contamination
An Isolation Forest parameter specifying the expected proportion of anomalies in the dataset. It determines the threshold used by `predict()` to classify points as Normal or Anomaly. The GA searches contamination values in the range [0.01, 0.30].

### CORS (Cross-Origin Resource Sharing)
A security mechanism that allows or restricts web applications from making requests to a different domain. MorphGuard configures CORS to allow the dashboard frontend (default: http://localhost:3000) to access the API.

### Crossover
A genetic operator that combines genes from two parent chromosomes to produce offspring. MorphGuard uses single-point crossover, where a random split point divides each parent and the halves are swapped to create children.

---

## D

### DataCleaner
The class in `feature_engineering/data_cleaner.py` responsible for preprocessing extracted features, including handling missing values, type casting, and infinite value replacement.

### DataNode
A component of the Hadoop Distributed File System responsible for storing data blocks. DataNode log entries form the majority of events in the HDFS log dataset.

### Decision Function
A method on the Isolation Forest model that returns the raw anomaly score for each data point. In MorphGuard, the decision function output is negated so that higher values indicate more anomalous behavior.

### Django Parser
A parser implementation (`parser/django_parser.py`) for Django web framework access logs. Demonstrates the extensibility of the parser architecture to non-HDFS log formats.

---

## E

### Elitism
A GA strategy that ensures the best individual from the current generation is carried forward to the next generation unchanged. MorphGuard uses an elitism count of 1, guaranteeing the best-known solution is never lost.

### Ensemble Method
A machine learning technique that combines multiple models to produce better predictions than any single model. Isolation Forest is an ensemble of isolation trees.

### Event Entropy
Shannon entropy computed over the distribution of event types within a block's lifecycle. Higher entropy indicates more diverse event types, while lower entropy suggests a more predictable (or limited) event pattern.

### Event Type
A classification assigned to each log entry based on keyword matching in the content field. Event types include RECEIVING_BLOCK, RECEIVED_BLOCK, SERVING_BLOCK, REPLICATION, DELETION, BLOCK_MAP, PACKET_RESPONDER, WRITE_BLOCK, ALLOCATE, VERIFICATION, STATE_CHANGE, and OTHER.

---

## F

### F1 Score
The harmonic mean of precision and recall: F1 = 2 * (Precision * Recall) / (Precision + Recall). F1 is the primary optimization metric in MorphGuard because it balances the ability to detect anomalies (recall) with the accuracy of those detections (precision), which is essential given the class imbalance.

### Feature Engineering
The process of transforming raw parsed log data into numerical features suitable for machine learning models. In MorphGuard, features are computed at the block level and include event counts, temporal statistics, error indicators, and distributional measures.

### Feature Extractor
The class in `feature_engineering/feature_extractor.py` that performs block-level aggregation and feature computation from parsed log DataFrames.

### Feature Selection
The process of choosing which features to include in a machine learning model. In MorphGuard, feature selection is performed by the GA using binary genes that turn features on (1) or off (0).

### Fitness Function
A function that evaluates how good a candidate solution is in a GA. In MorphGuard, the fitness function trains an Isolation Forest with the chromosome's parameters and returns the F1 score on the Anomaly class.

### Flask
A lightweight Python web framework used by MorphGuard to implement the REST API. Flask provides routing, request handling, and blueprint-based modular application structure.

---

## G

### Genetic Algorithm (GA)
An evolutionary optimization algorithm inspired by natural selection. A population of candidate solutions (chromosomes) evolves through selection, crossover, and mutation over multiple generations, with the fittest individuals having a higher probability of contributing to the next generation.

### Generation
One iteration of the GA evolution cycle: fitness evaluation, parent selection, crossover, mutation, and replacement. MorphGuard tracks the best fitness per generation to monitor convergence.

---

## H

### HDFS (Hadoop Distributed File System)
A distributed file system designed for large-scale data storage across clusters of commodity hardware. HDFS logs record block operations (read, write, replicate, delete) and form the primary dataset for MorphGuard's anomaly detection.

### HDFSParser
The parser implementation in `parser/hdfs_parser.py` that extracts structured fields from HDFS log lines using regular expression pattern matching.

### Hugging Face
A platform for sharing machine learning models and datasets. MorphGuard hosts its HDFS log dataset on Hugging Face for convenient distribution.

### Hyperparameter
A parameter of a machine learning model that is set before training begins (as opposed to parameters learned during training). Isolation Forest hyperparameters include n_estimators, max_samples, max_features, and contamination.

---

## I

### Isolation Forest
An unsupervised anomaly detection algorithm that isolates anomalies by randomly partitioning the feature space using an ensemble of decision trees. Anomalous points require fewer partitions (shorter path lengths) to be isolated, making them distinguishable from normal points.

### Isolation Tree
A single tree in the Isolation Forest ensemble. Each tree recursively partitions data by randomly selecting a feature and a split value within that feature's range.

---

## J

### joblib
A Python library used for serializing (saving) and deserializing (loading) trained scikit-learn models. MorphGuard saves Isolation Forest models as `.pkl` files using joblib.

### JWT (JSON Web Token)
A compact, URL-safe token format used for authentication. MorphGuard issues JWTs upon login/registration with a 24-hour expiration. The token contains the user's ID and email as claims.

---

## M

### Model Evolution
The process of comparing a newly trained candidate model against the current production model and promoting it if it demonstrates superior performance. MorphGuard tracks model versions, promotion decisions, and metric improvements.

### Model Versioning
Sequential numbering of production models (v1, v2, v3, ...). Each version has associated model file, metrics file, and promotion metadata stored in `current_version.json`.

### MongoDB
A NoSQL document database used by MorphGuard for storing user accounts and upload metadata. MongoDB's flexible schema is well-suited for the varied document structures used across collections.

### MorphGuard
The name of the system, reflecting "morphing" behavior (continuously evolving models via GA) in service of "guarding" cloud infrastructure through anomaly detection.

### Mutation
A genetic operator that randomly modifies genes in a chromosome to maintain population diversity and explore new regions of the search space. MorphGuard mutates 20% of genes per chromosome.

---

## N

### NameNode
The master node in HDFS responsible for managing the file system namespace and regulating access to files. NameNode log entries appear in the HDFS dataset alongside DataNode entries.

---

## P

### Path Length
In an Isolation Tree, the number of random partitions required to isolate a data point. Shorter path lengths indicate anomalous points (they are easier to isolate because they are few and different).

### Precision
Of all blocks predicted as Anomaly, the proportion that are actually anomalous. High precision means few false alarms.

### PyGAD
A Python library providing Genetic Algorithm implementations. MorphGuard uses PyGAD to evolve candidate model configurations.

### PyMongo
The official Python driver for MongoDB. Used by MorphGuard via the `db.py` module for all database operations.

---

## R

### Recall
Of all blocks that are truly anomalous, the proportion that the model successfully detects. High recall means few missed anomalies.

### Random State
A fixed seed value (42 in MorphGuard) used to initialize random number generators, ensuring reproducible results across runs.

---

## S

### SIEM (Security Information and Event Management)
A category of security tools that collect, aggregate, analyze, and report on security-relevant data from across an organization's IT infrastructure. MorphGuard is described as "SIEM-inspired" because it implements core SIEM capabilities (log ingestion, anomaly detection, alerting) with evolutionary optimization.

### Stratified Sampling
A sampling technique that preserves the proportion of each class in the sample. In MorphGuard, stratified subsampling ensures the GA's fitness evaluation dataset maintains the ~2.9% anomaly rate of the full dataset.

### Steady-State Selection (SSS)
A parent selection method in GAs where only a portion of the population is replaced each generation, preserving diversity while maintaining selection pressure toward fitter individuals.

---

## T

### Token Required
A Flask decorator (`@token_required` from auth.py) that validates the JWT bearer token on protected endpoints. On success, it injects the decoded token claims into `g.user`.

### Training Feature Columns
The subset of extracted features used as input to the Isolation Forest model. Defined in `detection/isolation_forest.py` as all `FEATURE_COLUMNS` minus `EXCLUDED_FEATURE_COLUMNS` (currently excluding `is_error`).

---

## U

### Unsupervised Learning
A machine learning paradigm where models are trained without labeled data. The Isolation Forest in MorphGuard is trained unsupervised — anomaly labels are only used for evaluation, never for training.

---

## V

### Version History
A JSON log (`version_history.json`) recording every model promotion and rejection decision with metadata including metric values, timestamps, and file paths. Provides an audit trail for security compliance.

---

## W

### Werkzeug
The WSGI toolkit that underlies Flask. Werkzeug provides request/response handling, routing, and a development server. Its HTML debug page is intentionally replaced by JSON error responses in MorphGuard to maintain CORS compatibility.

---

*This glossary is part of the MorphGuard project documentation.*
