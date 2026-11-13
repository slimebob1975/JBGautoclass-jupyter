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

Buttons that perform an action are green when they can be used and red,
dimmed, while there is nothing to do yet, for example **Save** before
anything has changed. Hold the pointer over a red button to see why. Orange
buttons can be used but cannot be undone, such as **Delete**. On long
pages, the page's main button is at both the top and the bottom.

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

**Delete the project…** deletes the project with everything in it:
datasets, configurations, runs with their results and predictions, and
models. Type the project's name to confirm; the **Delete** button turns
orange when the name is right. Tables in SQL Server are not touched.

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
on. A training or mixed dataset can switch to the other role with **Change
role to …**; its column roles and validation are kept. Training uses only
rows with a class, and a mixed dataset can also be predicted on the
Prediction tab. Prediction datasets cannot switch role. Below the dataset are the views Preview, Profile, Classes and Column
roles.

**Delete the dataset…** deletes the dataset with its column roles, every
version of its CSV file (corrected ones included), the training runs on it,
the models trained on it and the predictions made on it or with those
models. Configurations stay: they are not tied to a dataset.

### Column roles

The column roles belong to the training dataset and apply to every run with
it. Exactly one column must be the ID column and exactly one the class
column, and at least one column must be a feature: a numeric feature, a text
feature or a categorical feature. Ignored columns are not used.

In SQL datasets the ID column must be a whole-number column (int, bigint,
smallint or tinyint), not for example uniqueidentifier or text. Otherwise
**Treated as** shows a warning, validation fails the ID column and a run does
not start. In CSV datasets the ID column can hold anything.

A date column can have the role **Date feature**. It then becomes the year
(a number), month and weekday (points on a circle, so that December lies
next to January) and a mark for a missing date. Dates are never suggested
automatically: a date column is suggested as ignored, and **Treated as**
shows that it can be a date feature. Dates must be in the format YYYY-MM-DD,
optionally with a time; in SQL tables the types date, datetime, datetime2 and
smalldatetime count. Keep in mind that the year only describes the period
the model was trained on; a new year is a value it has not seen.

At most one column can be the **Group column**, for example a person's
identity. It keeps its rows together when data is split, so that one group
is not in both training and test, but it is not used as a feature. It is
needed when the configuration keeps groups apart. Prediction datasets must
also contain the group column.

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

On the left, below the list, **Pipeline** shows the steps of the open
configuration in the order they run. A green step is used, a purple one
compares several alternatives and a dashed one is not used. The steps tried
together in cross-validation are framed, with the number of combinations.
Click a step to open its section. The picture shows the saved version.

In **Data split**, choose whether groups are kept apart in cross-validation
or in the holdout split. The group column is not set here but as a column
role on the Dataset tab; the section shows which column is used. Keeping
groups apart in the holdout split needs two classes. In **Oversampling** and **Undersampling**, tick the
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
  It is red when the configuration is already chosen.
- **Delete the configuration…** deletes it with every version, the training
  runs made with it, their models and the predictions made with those
  models (see Deleting below).
- CSV runs only train. New configurations in CSV projects therefore have
  "Predict unlabelled rows" unticked.

## Training

The list **Before the run** shows whether everything needed is there: an
active training dataset with column roles, a chosen configuration and, for
SQL, a sign-in. For a point that is not met, **Put right** leads to where it
is fixed. **Start training** starts the run with the active dataset
and the chosen configuration.

A finished run can be deleted with **Delete the run…** (also on the
Prediction and Results tabs).

The progress updates itself. While the models are trained and compared, a
thinner bar shows how many algorithm combinations are done, for example
"12 of 40". The run continues even if you close the
browser, and it can be stopped with **Cancel the run**. **Show log** shows
the run's log.

**Cancel the run** asks the run to stop: it stops after the step it is
working on, such as the algorithm combination being tried. A long step can
take a while. Once you have cancelled, **Stop now** ends the run at once; the
step in progress is lost, which does not matter for a cancelled run. If the
program on the server is stopped or ends unexpectedly, its runs are marked
interrupted: at once when the server notices, otherwise after five minutes.

If the configuration also predicts rows without a class but the training
dataset has no such rows, the run only trains and skips the prediction; the
log says so.

When the run has finished, the evaluation on the holdout data, the
confusion matrix and the most important features are shown.

## Results

The project's finished training runs. For a selected run there are the
views Overview, Model comparison, Mispredictions, Feature importance and
Dark Numbers. Class names and row values are shown only to the person who
started the run. Older runs may lack some details.

**Dark Numbers** shows, when the configuration estimates Dark Numbers, the
estimated share of hidden cases for each target class. Four estimates are
made: with the final model and with the cross-trained model on all data, a
combined one, and with the cross-trained model on the test data. They are
expected to grow in that order and together form a heuristic range; the
view says whether the order holds. Below are the calculations and the
confusion matrices.

In **Mispredictions** of an SQL run you can correct classes that are wrong
in the source data. Choose the **Right class** for the rows, then **Review
the corrections**. A page shows the changes, and only when you choose
**Write the corrections** are they written to the source table with your SQL
sign-in, all at once or none. The sign-in must be allowed to change the
table. Corrected rows are marked, and the corrections are listed below the
table; **Undo** writes the old class back. The next training uses the
corrected classes.

For a CSV run it works the same way, but the corrections are saved in a new
version of the dataset's file, which every later run reads; the uploaded file
is kept. The dataset needs validating again. **Download CSV** on the Dataset
tab downloads the file with the corrected classes.

In **Dark Numbers** each estimate is also shown in percent, with a bar on
the target class's scale, and below the table the range from the smallest
to the largest estimate.

Wide tables have a scrollbar both below and above the table.

## Models

The models the training runs have created, with where they came from and
their key figures, and the model's remaining runs. **Choose model** makes an
available model the one used for prediction. **Delete the model…** deletes
it with its files and the prediction runs made with it.

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
can be downloaded as a file with **Download the predictions**. In CSV
projects, **Download with the original columns** downloads the predictions
together with all columns of the predicted rows, from the file the prediction
read.

## Run History

Available in a later version.

## Good to know

- Logs and predictions may contain values from your data and are shown only
  to the person who started the run.
- Every run keeps its results in a folder of its own on the server.
- **Deleting**: what owns something takes it with it. A run takes its log
  and result files; a model its files and its prediction runs; a
  configuration its versions, its training runs and their models; a
  dataset its file versions, the runs on it and the models trained on it; a
  project everything in it. CSV file versions that nothing uses any more are
  cleaned up when a dataset is deleted. A page
  first lists exactly what is deleted and what stays; **Delete** (orange)
  carries it out and cannot be undone. Nothing is deleted while a run that
  would go is queued or running. When a training run is deleted but its
  model is kept, only the run's settings are kept for the model, hidden.
  Tables in SQL Server are never changed: predictions and corrections
  already written stay.
- This tab shows the program's version.
