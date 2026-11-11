# JBG Automatic Classification

The program trains machine learning models that sort the rows of a table
into classes. It learns from rows that already have a class and can then
suggest a class for rows that lack one. Data come either from a table in SQL
Server or from a CSV file.

When training, the program compares several algorithms and combinations of
preprocessing, evaluates them on data kept out of the training and saves the
best model. The model can then be used for prediction, that is, to classify
new rows.

## The work order

The tabs follow the work order from left to right:

1. **Projects**: create or open a project.
2. **Dataset**: add data, set the column roles and validate.
3. **Configuration**: choose the machine learning settings.
4. **Training**: start a training run and follow it.
5. **Results**: review how good the model turned out.
6. **Models**: choose the model to use.
7. **Prediction**: classify rows without a class.

The line below the title shows the current work: the open project, the
active training dataset, the configuration chosen for runs and the chosen
model. The program opens on the Overview tab, and the title "JBG Automatic
classification" leads there.
Choose the language with SV/EN at the top right.

## Overview

A summary of the open project: the active training dataset, the chosen
configuration, the latest training and prediction runs and the project's
models. Each part links to the tab where it is changed. Without an open
project the page is empty.

## Projects

All work belongs to a project. Enter a name and a source (SQL or CSV) and
choose **Create project**. The source cannot be changed afterwards. Select
the project in the list and choose **Open project** to work with it. The
cross by the project name in the line below the title closes the project.

## Dataset

A project can have several datasets. Each dataset has a role: Training,
Prediction or Mixed.

- **CSV projects**: choose a file (at most 25 MB, with a header row; the
  delimiter is detected automatically), enter a name and a role and choose
  **Upload**.
- **SQL projects**: sign in with a user name and password or with Windows
  authentication. Choose a catalog and a role, **Show tables**, choose a
  table, enter a name and choose **Register**. The sign-in is kept only in
  the server's memory; after a server restart, sign in again.

Choose **Activate for training** on the dataset the model is to be trained
on. Below the dataset are the views Preview, Profile, Classes and Column
roles.

### Column roles

The column roles belong to the training dataset and apply to every run with
it. Exactly one column must be the ID column and exactly one the class
column, and at least one column must be a feature: a numeric feature, a text
feature or a categorical feature. Ignored columns are not used.

If the dataset has no saved column roles, the form is filled in with
suggestions: the first filled-in column with a different value on every row
becomes the ID column, numbers become numeric features, text with at most as
many distinct values as the categorical limit becomes categorical features
and other text text features. Empty and single-valued columns are ignored,
as are, for example, date columns in SQL tables. The class column is not
suggested: choose it in the form, or show a column's distribution on the
Classes view, and it becomes the class column in the suggestion. The
suggestion is not kept until you choose **Save column roles**.

**Categorical limit** (default 30, allowed 2–1000): text columns with at most
this many distinct values are treated as categories, other text columns as
free text. Categorical features are always categorised. The column "Treated
as" shows the outcome.

Save with **Save column roles**, then validate the dataset again with
**Validate dataset** at the bottom.

If the project already has a dataset with the same columns, choose it under
**Copy column roles from** and **Fill in**. The roles are filled in but not
kept until you save; columns that are missing or not numbers are pointed
out. A configuration can be used with every dataset in the project.

## Configuration

A configuration holds the machine learning settings. **Create
configuration** starts with the default settings.

The sections come in the order the steps run: Run type, Text processing,
Feature selection, Data split, Scaling, Oversampling, Undersampling,
Dimensionality reduction, Algorithms, Decision, Evaluation, Dark Numbers
and, in SQL projects, Predictions, last Advanced. Scaling, sampling,
dimensionality reduction and the algorithm are tried together in
cross-validation. Every section is closed when the page opens; click a
section's title to open or close it. Sections that were open when you saved
stay open after saving. Less common settings are under **Advanced settings**
in each section.

In **Data split**, the **Group column** is chosen among the features of the
active training dataset, or **None**. It is used only when groups are kept
apart in cross-validation or in the holdout split. A saved column that is
not a feature is marked. In **Oversampling** and **Undersampling**, tick the
methods to try; with nothing ticked, no sampling is used.

In **Dark Numbers**, the **Target class** is chosen among the classes of the
active training dataset's class column, or **All classes**. The classes are
read when the section is opened (SQL projects need a sign-in). Below the
fields is the formula of the chosen method and alpha, with the target class
and the fallback; it follows your choices at once, and alpha "Separated" can
only be chosen with the method "Linear".

- The **All** box in a list ticks every choice. It is also ticked when the
  last choice is ticked, and it is unticked when a choice is unticked.
- **Save configuration** creates a new version; earlier versions are kept.
- **Choose for runs** makes the configuration the one used for training.
- CSV runs only train. New configurations in CSV projects therefore have
  "Predict unlabelled rows" unticked.

## Training

The list **Before the run** shows whether everything needed is there: an
active training dataset with column roles, a chosen configuration and, for
SQL, a sign-in. **Start training** starts the run with the active dataset
and the chosen configuration.

The progress updates itself. While the models are trained and compared, a
thinner bar shows how many algorithm combinations are done, for example
"12 of 40". The run continues even if you close the
browser, and it can be stopped with **Cancel the run**. **Show log** shows
the run's log.

If the configuration also predicts rows without a class but the training
dataset has no such rows, the run only trains and skips the prediction; the
log says so.

When the run has finished, the evaluation on the holdout data, the
confusion matrix and the most important features are shown.

## Results

The project's finished training runs. For a selected run there are the
views Overview, Model comparison, Mispredictions and Feature importance.
Class names and row values are shown only to the person who started the
run. Older runs may lack some details.

## Models

The models the training runs have created, with where they came from and
their key figures. **Choose model** makes an available model the one used
for prediction.

## Prediction

Prediction uses the model chosen on the Models tab.

- **SQL projects**: predict the training dataset's rows without a class, or
  a registered prediction dataset that is compatible with the training
  dataset. The predictions are written to the prediction table in SQL
  Server. A table without a class column is predicted in full.
- **CSV projects**: predict the training dataset's rows without a class, or
  a prediction (or mixed) dataset with the same columns. The rows without a
  class are predicted; a file without a class column is predicted in full.
  If the file does not fit the model, or every row already has a class, the
  tab says so before anything starts.

When the run has finished, the predictions are shown page by page, and they
can be downloaded as a file with **Download the predictions**.

## Run History

Available in a later version.

## Good to know

- Logs and predictions may contain values from your data and are shown only
  to the person who started the run.
- Every run keeps its results in a folder of its own on the server.
- This tab shows the program's version.
