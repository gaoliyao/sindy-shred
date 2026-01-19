The @sst_sindy_shred.ipynb and @sindy-shred_functionalized.toy-data.v2.ipynb contain our SINDy-SHRED running pipeline. Keep the same data geneation, model definition, analysis, equation learning and graphs, and improve the structure and documentation of the notebook. Make changes with the creation of two new factored notebooks @sst_sindy_shred_refactor.ipynb and @sindy-shred_functionalized.toy-data.v2_refactor.ipynb. 

Notice that an important change is right now we have the @driver.py function. This driver function is important when we use it to run the code. It will simplify everything from @sst_sindy_shred.ipynb. 

Review the existing notebook and python file and identify:
- What data is generated and how
- What is the data loading from SHRED
- Which model are we using and what are the hyperparameters
- What later analysis we get from SINDy learning
- What graphs are plotted and what they represent

**Refactoring Requirements**

1. Notebook Structure & Documentation
    - Add proper documentation and markdown cells with clear header and a brief explanation for the section
    - Organize into logical sections:

2. Code Quality Improvements
   - Create reusable functions with docstrings
   - Implement consistent naming and formatting

3. Refactor the Python file for @driver.py and @models.py and @sindy_shred.py. This file should serve as a single entry point for all model calling and related function data calling (data pipeline only, no generation). The purpose is to make @sst_sindy_shred.ipynb and @sindy-shred_functionalized.toy-data.v2.ipynb all notebooks easy to use. Right now, @driver.py should have everything implemented, we need to: 
    - change the name from driver uniformly to sindy_shred, and since we already have SINDy_SHRED for model, make that into SINDy_SHRED_network instead. Make sure this kind of change uniformly happens, so that we don't have driver. 
    - Make sure to create two new notebooks (refactored) as @sst_sindy_shred_refactor.ipynb and @sindy-shred_functionalized.toy-data.v2_refactor.ipynb and make them align and cleaned up. And use the new sindy_shred module code.  
    - Also consider having a unified plotting tool to enable simplified plotting in the notebooks. 

**Deliverables Expected**
- Refactored Jupyter notebooks with all improvements
- Updated python file with name changed, simplified, but keep all the original functions. 

**Success Criteria**
- Easy-to read code & notebook (do not use icons in the printing statements or markdown cells)
- Reusable code that can be applied to future datasets
- Maintainable structure that other analysts can easily understand and extend
- Maintain all existing analyses while improving the quality, structure, and usability of the notebook.
                                                                                                    