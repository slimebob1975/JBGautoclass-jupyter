THE README INSTRUCTION FILE FOR IAFautoclass_GUI.ipynd

To use the Jupyter GUI for IAF Python autoclassification script, do as follows:

1. Get a hold of the IAFautoclass GUI codes in exactly one of the two following ways:
    a) Use GIT:
        i) Install GIT, see https://github.com/git-guides/install-git 
        ii) Clone the code from the repository in a terminal window: 
                git clone https://github.com/slimebob1975/IAFautoclass-jupyter
            into an appropriate directory
    b) Ask a friend to get a copy of the code

    There are two versions of this. The legacy (if wanted) is tagged v1.0, and the main branch is version 2.

2. Copy the file example.env to a new file called .env and change
       the variables such that they reflect your SQL Server environment
3. Download and install Anaconda from: https://www.anaconda.com/products/distribution

4. Start Anaconda Navigator and launch Jupyter-lab from within

5. In the file explorer window to the left in Jupyter-lab, browse your way to the file
    IAFautoclass_GUI.ipynb
6. If you see the IAF logo and some webb-like widgets in the right hand side, then all is ok!
    
Notice:
* The connection to SQL Server uses integrated security, so you will only be able to see the
databases and datatables that also show up in e.g. Microsoft SQL Server Management Studio
    
Troubleshooting:
* If you only see text in the right hand side window of Jupyter-lab, try to restart the kernel
(push the double play symbol)