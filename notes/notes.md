# 1. Introduction to Data Warehousing and Data Mining
## Overview of Introduction 
* Data mining: discovering interesting patterns from large amounts of data
* Natural evolution of db technology
* KDD process involved data cleaning, data integration, data selection, transformation, data mining, pattern evaluation and knowledge presentation
* Data mining functionalities: characterisation, discrimination, association, classification, clustering, outlier and trend analysis
* Data mining systems and architectures
* Major issues in data mining: Different types of data, performance, different DM languages, 
## Data Mining
* Extraction of interesting patterns from huge amounts of data
* Searching for patters of interest
* Also known as:
  * knowledge discovery in databases (KDD)
  * knowledge extracton
  * data analysis
  * information harvesting
## Potentional Application
* Data analysis and decision support
  * Market analysis and management
    * This include specifically targeting customers, market basket analysis and market segmentation etc
  * Risk analysis and management
    * Forecasting, customer retention, improved underwriting, quality control and competitive analysis
* Other Applications
  * Text miining (ie. reddit, email) ~ Web mining for sentiment analysis etc
  * Stream data mining (ie. twitter, live flow of information that's constantly incoming)
# Market Analysis and Management
* Where do we get this data from?
  * Cred card transation
  * Loyality cards
  * Discount coupons
  * Customer complaint calls
  * Lifestyle studies
* Target marketing
  * Finding clusters of "model" 
  * Determine customer purchasing patterns over time
* Cross-market analysis
  * Association/co-relations between product & prediction based on such association
* Customer profiling
  * What types of customer buy what products (clustering or classification)
* Customer requirement analysis
  * Identifying the best products for different cusotmers
  * Predict what factors will attract new customers
* Provision of summary information
  * Multidimensional summary reports
  * Statistical summary infomation
## Knowledge Discovery Process
![KDD Process](assets/kdd.png)
* Learning the application domain
  * Relevant prior knowledge and goals of application
* Data Selection: Creating a target data set
* Data Cleaning and Preprocessing:
* Data Reduction and transformation
  * Finding useful features, dimenisonality/variable reduction and invarient representatoin
* Choosing functions of data mining
  * Summarization, classification, regression, association and clustering
* Choosing the mining algoirthms
* Pattern evaluation and knowledge presentation
## Data Mining and Business Intelligence
![Data Mining and Busienss Intelligence](assets/dmbi.png)
## Architecture - Typical Data Mining System
![Architecture](assets/dmarchi.png)
## Kinds of Data
* Relational Database
* Data Warehouse
* Transactional Database
* Advanced database and information repository
  * Object-relational database
  * Spatial and temporal data
  * Time-series data
  * Stream data
  * Multimedia database
  * Heterogeneous and legacy database
  * Text databases & WWW
## Data Mining Functinalities
* Cocept Description: Characterization and discriminiation
  * Generalize, summarize and contrast characteristics
* Association (correlation and causality)
  * Diaper -> Beer [0.5%, 75%]
* Classification and Prediction
  * Construct models that descripbe and distinguish classes / concepts for future prediction
    * Eg. Classify countries based on climate or classify cars based on gas mileage
  * Presentation: decision-tree, classification rule, neural network
  * Predict some unkown data
* Cluster Analysis
  * Class label is unknown. We group data so we can form new classes
  * Maximize intra-class similarity & minimizing interclass similarity
* Outlier analysis
  * Outlier: a data object that doesn't comply with the general behaviour of the data
  * Might be in the form of noise or excpetion. Not useful for insights
* Trend and evaluation analysis
  * Trend and deviation, ie. regression analysis
  * Sequential pattern mining, periodicity analysis
  * Similarity-based analysis
## Patterns in data
* DM may produce lots of patterns, but not all of them are useful.
* Interstingness Measures
  * A patter that's interesting and easily understood by humans, valid on new or test data with some degress of certainty or some hypothesis that the user is trying to confirm
* Subjective vs Objective Interestingness Measures:
  * Subjective: Basd on the user's belief in the data.
  * Objective: based on statistics and structs of patterns.
* Completeness
  * Finding all the intersting patterns
  * Can a DM system find all the interesting patterns?
  * Heuristic based vs. exhaustive search
  * Association vs Classification vs Clustering
* Optimisation Problem
  * Search for only intersting patterns
  * Can a DM system find only the intesting patterns?
  * Method:
    * Generate all the patterns and then filter out the unintersting ones
    * Mining Query Optimization: Generate only the intersting patterns 
## Classification Schemes
* General Functionality
  * Descriptive data mining
  * Predictive data mining
* Different views, different classifications
  * Kinds of data to be mined
  * Kinds of knowledge to be discovered
  * Kinds of techniques utilized
  * Kinds of applications adapted 
## Multi-Deimension View of Data Minig
* Data to be mined
  * Relational, data warehouse, transactional, stream, object-oriented/relational, active, spatial, time-series, text, multi-media, heterogeneneous. legacy, webpages
* Knowledge to be mined
  * Characterization, discrimination, association, classification, clustering, trend/deviation, outlier analysis
* Techniques Utilised
  * Database-oriented, data warehouse (OLAP), Machine Learning, Statistics, Visualisations
* Applications Adapted
  * Retail, telecommunication, banking, fraud analysis, bio-data mining, stock market analysis, web mining
## Issues in Data Mining
* Mining Methodolody
  * Mining knowledge in different sort of formats: biodata, stream, web etc
  * Performaces: efficiency, effectiveness and scalability
  * Pattern evaluation
  * Incorporation of background knowledge
  * Handling noise and imcomplete data
  * Parallel, distributed and incremental mining methods
  * Knowledge Fusion: Integration fo discoveryed knowledge with existing knowledge
* User Interaction
  * Data mining query languages and ad-hoc mining
  * Expression and visualization of data mining results
  * Interactive mining of knowledge at multiple levels of abstraction
* Applications and social impacts
  * Domain-specific data mining & invisible data mining
  * Protection of data security, integrity and privacy


# 2. Data Warehouse and OLAP
## Overview of Data Warehouse and OLAP
## Data Analysis Problems
* Redundant data found across departments
* Hetergeneous sources
  * Relations DBMS
  * Unstructured data in files (eg. MS Excel) and in documents (MS Word)
* Data is suited for different operating systems 
  * Doesn't integrate well across departments
* Bad data quality
  * Mising data, imprecise data etc
* Data is volatile
  * Data deleted in operating systems
    * Data changes over time - no historical information
## Data Warehouse
* Process of constructing and using data warehouses
* **Data mart** subset of a data warehouse that's usually specific to particular business department. Data warehouses are made up of integrated data marts.
* Is subject-oriented, integrated, time-variant and non-volatile collection of data in support of management's decision making process
### Subject Oriented
* Provide a clear view around the particular subject
  * ie. don't include data that's not relevant in the decision support process
* Organised around major subjects such as customer, product, sales etc
### Integrated
* Integrating multiple heterogeneous data sources 
  * Relational DBs, Flat files etc
* Data cleaning and data integration techniques are applied
  * Ensure consistency in naming conventions, encoding structures, attribute measures, etc. among differnt data sources
  * When data is moved to the warehouse, it's normalised
### Time Variant
* Time horizon for data warehouse is significantly longer than that of operational systems
  * Operational Database: current value data
  * Data warehouse data: provide information from a historical perpective
* Every key structure in the data warehouse
  * Contains an element of time, explicity or implicitly
  * The Key of operational data may or may not contain time element.
### Non-Volatile
* Once data is in the DW it will not change. So historical data in a data warehouse should never be altered.
* A physically separate store of data transformed from the operational environment
* Operational update of data doesn't occus in the data warehouse environment. Does not require transaction processing, recovery, and concurrency control mechnisms
* Requires only two operations in data accessing:
  * Initial loading of data
  * Access of data
## Data Warehouse Architecture
* Extract data from different data sources
  * Clean / Transformation
* Bulk Load / Refresh
  * Warehouse is offline
* OLAP-server provides multidimensional view
* Multidimensional-OLAP
* Relational-OLAP
![DW Architecture](assets/dwarchi.png)
## Separate Data Warehouse?
* High performance for both systems
  * DBMS: tuned for OLTP: acess methods, indexing, concurrency control, recovery
  * Warehouse: tuned for OLAP: complex OLAP queries, multidimensional view, consolidation
* Different functions and different data:
  * Missing data: decision support requres hostrical data which normal DBs don't maintain (ie. every new time-period a frame is created)
  * Data consolidation: decision support requires consolidation (ie. aggregation and summarisation) from heterogeneous sources
  * Data quality: different sources usually have inconsistent representations, codes and formaats which have to be reconciled.
## OLAP Servers
* Present business users with multidimensional data from data warehouses or data marts, without regarding how the data is stored.
* Different workloads (OLTP vs OLAP)
* Queries hard/infeasible for OLTP, e.g,
  * Which week we haved the largest amount of sales
  * Does the sales of diary products increase over time?
  * Generate a spreadsheet of total sales by state and by year.
* Difficult to represent these queries by using SQL
## OLTP vs. OLAP
![OLAP VS OLTP](assets/olap-vs-oltp.png)
## Databases vs Data Warehouses
![DB VS DW](assets/db-vs-dw.png)
## The Multidimensional Model
* Datawarehouse is based on a multidimensional data model which views data in the form of a **data cube**, which is a multidimensional generalisation of 2D spreadsheet
* **Data cubes** are modeled using dimensions and facts.
* **Facts**: the subject it models. Facts are numerical measures. 
* **Dimensions**: context of the measures
* **Measures**: numeric function that can be evaluated at each point in the data cube space.
  * Distributive: affregate function that can be computed in a distributed manner as follows. 
* **Hierarchies**: provides contexts of different granularities (aka. grains)
* Goals for dimensional modeling:
  * Surround facts with as much relevant context (dimensions) as possible
### Example
* Subject: analyze total sales and profits
* Fact: Each Sales Transaction
  * Measure: Dollars_Sold, Amount_Sold, Cost
  * Calculated Measure: Profit
* Dimensions:
  * Store
  * Product
  * Time
## Visualizing the Cubes
* 2D view of sales data for a "AllElectronic" according to time and item.
  - ![2D view of sales](assets/dc-2d.png)
* 3D view of sales data for a "AllElectronic" according to time, item & location.
  - ![3D view of sales](assets/dc-3d.png)
## 3D Cube and Hierarchies
- ![3D view of sales](assets/dc-3d-2.png)
* **Concept Hierarchy**: defines a sequence of mappings from a set of low-level concepts to higher-level, more general concepts. 
  - ![Concept Hierarchy](assets/concept-hierarchy.png)
## Cuboids
- ![3D view of sales](assets/lattice-of-cuboids.png)
* **Apex cuboid**  is the 0-D cuboid which holds the highest level of the summation. The apex cuboid is typicallly denoted by *all*. 
* A complete cube of d-dimensions consists of the product of  (L_i+1), from i=1 to d, where L_i is the number of levels (not including ALL) on the i-th dimension.
  * They collectively form the lattice.
  * **Full materialisation** refers to the computation of all the cuboids in the lattice defining a data cube.
    * Requires lots of storage space, especially when the dimensions increase.
    * The **curse of dimenstionality** refers to the excessive storage requirements, each with multiple levels.
  * **Partial materialisation** is where selective computation of a subset of the cuboids in the lattice. Ie. Isceberg cube is a data cube that stores only those cube cells that have an aggregate value (eg. count) about some minimum support theshold.

## Properties of Operations
* All operations are closed under the multidimensional model (i.e. both input and output of an operation is a cube).
* This means that they can be composed.
## OLAP Operations
* **Roll-up** (or drill-up): move up the hierarchy
  * This operation usually aggregates data in the hierarchy hence reducing the dimensionality. 
  * Eg. Hierarchy: street < city < state < country
  * Roll-up operation shoes the aggregates by ascending the location hierarchy from the level of city to the level of country (ie. rather than grouping the data by city, the resulting cube groups the data by country)
* **Drill-down**: move down the hierarchy
  * Reverse of roll-up. Introduces more hierarchies.
* **Slice and dice**: select and project one or more dimensional values
  * Slice performas a selection on one dimension of the given cube resulting in a subcube.
    * ![Slice](assets/OLAP_slicing.png)
  * Dice defines a subcube by performing a selection on two or more dimensions.
    * ![Dice](assets/OLAP_dicing.png)
* **Pivot (Rotate)**: aggregate on selected dimensions
  * Rotates the data aces in view to provide an alternative data presentation.
    * ![Rotate](assets/OLAP_pivoting.png)
## Logical Models
* Main approaches to represent these cubes using:
  1. Relational DB Technology
    * Start schema, snowflake schema, fact constellation  
  2. Multidimensional Technology
    * Just as multideminsional data cube
### Star Schema
* A fact table in the middle connected to a set of dimension tables
* Each dimension is represented by only one table, and each table contains a set of attributes.
  - ![Star Schema](assets/star.png)
### Snowflake Schema
* A refinement of star schema where some dimensional hierarchy is normalised into a set of smaller dimension tables, forming a shape similar to snowflake
  - ![Snowflake Schema](assets/snowflake.png)
### Fact constellation
* Multiple fact tables share dimension tables, viewed as a collection of stars therefore called galaxy schema or fact constellation
  - ![Fact constellation](assets/constellation.png)
## Query Language
* Two approaches:
  * Using DB tech: SQL (with extensions such as CUBE/PIVOT/UNPIVOT)
  * Using multidimensional technology: MDX
## Indexing OLAP Data: Bitmap Index and Join Index
* Efficient data accessing -> build an index
* **Bitmap indexing**: method used by OLAP servers to allow quick search in data cubes.
  * In the bitmap index for a given attribute, there's a distinct bit vector (Bv) for each value of v in the attribute's domain. 
  * Useful for low-cardinality domains becuse comparison, join and aggregation operations are reduced to bit arithmetic, thus reducing processing time.
  * ![Bitmap Index Example](assets/bmindex.png)
* **Join indexing**: register the joinable rows of two relations from a relational db.
  * Eg. Given the two relations R(RID, A) and S(B, SID) join on the attributes A and B, then join the index record which contains the pair (RID,SID). The join index can identify joinable tuples without performing costly join operations.
  * ![Join Index](assets/join-index.png)
## OLAP Server Architectures
### ROLAP
* Use relational DBMS to store and manage warehouse data and OLAP middleware to support missing pieves. ROLAP servers include optimisation for each DBMS back end, implementation of aggregate navigation logic and additional tools and services.
### MOLAP
* Support multidimensional data views through array-based multidimensisional storage arrays.
* Adopts a two-level storage representation to handle dense and sparse data sets. 
* Advantage: allows fast indexing to precomputed summarized data.
### HOLAP
* Hybring OLAP combines techniques used by ROLAP and MOLAP. 
* Benefits from greater scalability of ROLAP and faster computation from MOLAP.
  * Eg. Large volumes stored in ROLAP and aggregations kept in separate MOLAP store.
## Physical Model + Query Processing Techniques
* Issues
  1. How to store the materialised cuboids?
  2. How to compute the cuboids efficiently?
### ROLAP
#### Top-down Approach
* Involdes computing an cube by traversing down a multi-dimnensional lattice formed from the attributes in an input table.
* Begins by computing the frequent attribute value combinations for the attribute set at the top of the tree.
![Top-down approach](assets/tdCTree.gif)
### Botton-Up Computation (BUC) Approach
* Resource: http://www2.cs.uregina.ca/~dbd/cs831/notes/dcubes/iceberg.html
* The bottom-up computation algorithm (BUC) repeatedly sorts the database as necessary to allow convenient partitioning and counting of the combinations without lots of memory.
* Divide and conquer approach to compute the cube from the bottom up,
* Brief Algorithm:
  1. Counts the frequencies of the first attribute in the input table
  2. Partitions the database based on the frequent values of the first attribute, so tuples with frequent value for the first attribute are further examined
  3. Counts the combinations of values for the first two attributes and partions the database so that tuples that contain frequent combinaton of the first two attributes are further processed.
  4. Repeat for all attributes of the table
* ![BUC Tree](assets/BUCTree.gif)
### MOLAP
* Sparce array-based multidimensional storage engine
* Pros
  * Small size (good for dense cubes)
  * Fast indexing and query processing
* Cons
  * Scalability
  * Conversion from relation data 
* We use an injective mapping function (no same input maps to the same output) from cell to offset
* Example:
  * ![Example MOLAP Table](assets/molap-s0.png)
  1. Create mapping tables
     * ![Step 1 MOLAP Table](assets/molap-s1.png)
  2. An injective map from cell to offset
     * f(time,item) = 4time + item
     * ![Step 2 MOLAP Table](assets/molap-s2.png)
* Typically the multidimensional array is sparce, after sorting the final values according to the offset
  * Only need to store sorted slots, no need to store the offset.
### HOLAP
* Store all non-base cuboid in MD array
* Assign a value for ALL


# 3. Preprocessing 
## Overview of Preprocessing
## Why Preprocess
* Data can be dirty
  * **Incomplete**: attributes left blank
  * **Noisy**: containing outliers 
  * **Inconsistent**:  conmtaining discrepanies in data format
* Caused by when data is being collected
  * Different data sources
  * Human error
  * Software/hardware issues
* Why is it important?
  * Data mining quality depends on the quality of the dataset.
* Preprocessing is a critical step for data mining and makes up a majority of the work
## Major Tasks
* **Data Cleaning**: Filling in missing values, smoothing noisy data, identify or remove outliers and resolve inconsistencies
* **Data Integration**: Integration og multiple databases, data cubes or files 
* **Data Transformation**: Normalisation and aggregation 
* **Data Reduction**: Obtains reduced represention in volume but produces similar analytical results
* **Data Discretization & Data Type Conversion**
## Data Cleaning

### Missing Data
### Noisy Data
## Data Integration
### Handling Redundancy in Data Integration
## Data Transformation
## Data Reduction
### Dimensionality Reduction
### Data Compression
### Numerosity Reduction
### Discretization and Concept Hierarchy Generation


# 4. Classification & Prediction
## Overview of Classification
## Classification vs Prediction
## Classification and Regression
## Supervised vs Unsupervised Learning
## ML Terminology
## Two Step Process
## Decision Tree Classifier
## Overfitting in Classification
## DT Pruning Methods
## Pessimistic Post-pruning
## Classification in Large Databases
## Bayesian Classification
## Smoothing
## Text Classification
## Instance based learning
### KNN
## Lazy vs Eager Learning

# 5. Logisitic Regression 
## Overview of Logistic Regression
## Generative vs Discrimitive Learning
## Linear Regression
## Least Square Fit
## Minimizing a Function
## Least Fit Square for LR
## Probabilistic Interpretation
## Logistic Regression
## Learning W
## Understanding the Equilibrium
## Numeric Solution
## Gradient Ascent
## Newton's Method
## Regularisation
## Generalizing LR to Multiple Classes 

# 6. Hidden Markov Model
## Overview of HMM
## Applications
## HMM definition
## Markov Model
## Sequence Probability
## Generative Process
## 3 Problems
## Application - Typed Words
## Csting into Evaluation Problem
## Decoding Problem
## Join Probability
## Viterbi Algorithm

# 7. Support Vector Machine
## Overview of SVM
## Linear Classifiers
## SVM 
## Maximum Margin: Formalization
## Largest Margin
## Geometric Margin
## Help from Inner Product
## Derivation of Geometric Margin
## Linear SVM Mathematically
## Solving the Optimisation Problem
## Geometric Interpretation
## The Optimisation Problem Solution
## Soft Margin Classification
## Classification with SVMsLinear SVMs Summary
## Non-Linear SVMs
## The Kernel Trick
## Why Features Combinations?
## String Kernel
## Classication + SVM + Kernel
## Pros and COns of the SVM Classifier


# 8. Clustering
## Overview of Clustering
## What is Cluster Analysis
## General Applications of Clustering
## Examples of Clustering Applications
## What is Good Clustering
## Requirements of Clustering in DM
## Cluster Analysis
### Types of Data in CA
### Major Clustering Approaches


# 9. Spectral Clustering
## Overview of Spectral Clustering
## Quadratic Form
## Unnormalised Graph Laplacian
## Binary x induces a Clustering
## Min Cut vs Normalized Cut
## Connection to L
## Relaxation and Optimization 
## Spectural Clustering Algorithm Framework
## Notes on the Algorithm
## Comments on Spectral Clustering


# 10. Association Rules
## Overview of Association Rules
## What's Association Mining?
## Frequent Patterns and Association Rules
## Mining Association Rules
## Association Rule Mining Algorithms
## Apriori Property
## The Apriori Algorithm
## Generating Candidates in SQL
## Derive rules
## Bottleneck of Frequent-pattern Mining
## Notations and Invariants
## FP-tree
## FP Growth vs Apriori
