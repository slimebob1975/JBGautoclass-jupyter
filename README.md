<!--THE README INSTRUCTION FILE FOR JBG classification with an additional Jupyter GUI.-->
# JBG automatic spotchecking of classifiers

## Summary
This project aims on collecting many different data classifiers from several different
libraries under a common Jupyter GUI with the purpose of simplifying inital testing and 
spot-checking of algorithms for a particular dataset. For the time being JBG is limited
to a Windows and SQL Server environment and works for numerical, categorical and textual
data. It is partly a Building AI course project.

## How to use JBG
To use the Jupyter GUI for JBG Python autoclassification script, do as follows:

1. Get a hold of the JBGclassification GUI codes in exactly one of the two following ways:
    a) Use GIT:
        i) Install GIT, see https://github.com/git-guides/install-git 
        ii) Clone the code from the repository in a terminal window: 
                git clone https://github.com/slimebob1975/JBGautoclass-jupyter.git
            into an appropriate directory
    b) Ask a friend to get a copy of the code

    There are two versions of this. The legacy (if wanted) is tagged v1.0, and the main branch is version 2.

2. Copy the file example.env to a new file called .env and change
       the variables such that they reflect your SQL Server environment
       
3. Download and install Anaconda from: https://www.anaconda.com/products/distribution

4. Start Anaconda Navigator and launch Jupyter-lab from within

5. In the file explorer window to the left in Jupyter-lab, browse your way to the file
    JBGclassification_GUI.ipynb
6. If you see the JBG logo and some webb-like widgets in the right hand side, then all is ok!
    
Notice:
* The connection to SQL Server uses integrated security, so you will only be able to see the
databases and datatables that also show up in e.g. Microsoft SQL Server Management Studio
    
Troubleshooting:
* If you only see text in the right hand side window of Jupyter-lab, try to restart the kernel
(push the double play symbol)

== Terminal ==

To run the script in the terminal you need to have a file in `src\JBGclassification\config` with a functional config. 
Configs start with `autoclassconfig_` as the name, and will be saved when you create a new model using the GUI.

Go into `src\JBGclassification` and run `python JBGautomaticClassifier.py -f <path-to-file>`. The path to the file needs to
be on the format of `.config\filename.py`, so assuming that the config-file is `autoclassconfig_iris_abc0123.py` 
(check the `config` directory for the right name), the command is: `python JBGautomaticClassifier.py -f autoclassconfig_iris_abc0123.py`

=== Troubleshooting ===

1. You have to run the terminal command from the src\JBGclassification directory, due to imports and such
2. There are a lot of (unnecessary) warnings coming out of the 3rd-party libraries (in particular sklearn), which will clutter up
the terminal. To ignore them, use the `W` flag in the command (see below for usage)

==== W-flag ====

Source: https://docs.python.org/3/using/cmdline.html#cmdoption-W

To ignore all warnings: `python -Wi JBGautomaticClassifier.py -f <path-to-file>`

The full argument is `action:message:category:module:lineno`, and if you're targetting something "deeper" into the argument, leave any
intermediate things empty, IE `ignore::Classname` to target all warnings of Classname, no matter their message.

You can also put in as many of these specific warning-suppressions as you want, where for ease of use I'll write down the fine-grained
"most common" warnings below.

Specific warnings we know are irrelevant:
* UserWarning (-Wi::UserWarning)
* RuntimeWarning (-Wi::RuntimeWarning))

## Acknowledgments
This work was partly inspired by Jason Brownlee: https://machinelearningmastery.com/
