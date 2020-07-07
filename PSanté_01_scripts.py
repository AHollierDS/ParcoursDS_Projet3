# Preparing working environnement
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

# -----------------------------------------------------------------------------------------------------------------------------------------
def cut_pie(serie, cut_off = 0.05, title = None):
    cut_serie = serie[serie / serie.sum() > cut_off].copy().dropna()
    remain = pd.DataFrame(serie[serie / serie.sum() < cut_off].sum(), columns = ['Other']).T
    cut_serie = cut_serie.append(remain)            

    plt.figure(figsize = (8,8))
    plt.pie(cut_serie.iloc[:,0], autopct = lambda x: str(round(x,1)) + '%')
    plt.legend(cut_serie.index)
    plt.title(title, fontweight = 'bold')
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------
def retrieve_name(var):

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

# -----------------------------------------------------------------------------------------------------------------------------------------
def df_fillrates(df, col = 'selected columns', h_size = 15):
    """ Returns a barplot showing for each column of a dataset df the percent of non-null values """

    nb_columns = len(df.columns)
    df_fillrate = pd.DataFrame(df.count()/df.shape[0])
    df_fillrate.plot.barh(figsize = (h_size, nb_columns/2), title = "Fillrate of columns in {}".format(col))
    plt.grid(True)
    plt.gca().legend().set_visible(False)
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------
def drop_empty_col(df, min_value = 1):
    """ Removes columns in the df dataset if the columns contains less than min_value elements"""
    
    columns_list = df.columns.to_list()
    df_clean = df.copy()
    
    for col in columns_list:
        if df_clean[col].count() < min_value:
            df_clean = df_clean.drop(columns = [col])
   
    nb_col_i = len(columns_list)
    nb_col_f = len(df_clean.columns.to_list())
    
    print("{} columns out of {} have been dropped of the dataframe".format(nb_col_i - nb_col_f, nb_col_i))

    return df_clean

# -----------------------------------------------------------------------------------------------------------------------------------------
def list_criteria (df, criteria, h_size = 15):
    """ Returns the fillrate graph of columns in dataframe df whose name matches the criteria"""
    
    # Return the list of columns matching the criteria
    df_columns = pd.DataFrame(df.columns, columns = ['Column'])
    
    # Graph of fillrates on corresponding columns
    criteria_list = df_columns[df_columns['Column'].str.contains(criteria)]['Column'].to_list()
    df_fillrates(df[criteria_list], col = criteria, h_size = h_size)

# -----------------------------------------------------------------------------------------------------------------------------------------
def grade_distrib(df, size = (15,5)):
    
    # Grade series
    a_grade = df[df['nutriscore_grade']== 'a']['nutriscore_score']
    b_grade = df[df['nutriscore_grade']== 'b']['nutriscore_score']
    c_grade = df[df['nutriscore_grade']== 'c']['nutriscore_score']
    d_grade = df[df['nutriscore_grade']== 'd']['nutriscore_score']
    e_grade = df[df['nutriscore_grade']== 'e']['nutriscore_score']

    # Plotting
    plt.figure(figsize = size)
    plt.hist([a_grade, b_grade, c_grade, d_grade, e_grade], histtype = 'barstacked',
             color = ['green','lime','yellow','orange','red'],
             bins = 60, range = (-20, 40))

    # Plot parameters
    plt.xlabel('Nutrition score', fontstyle = 'italic', backgroundcolor = 'lightgrey')
    plt.ylabel('Number of products', fontstyle = 'italic', backgroundcolor = 'lightgrey')
    plt.title('Distribution of nutriscore and grade categorization', fontweight = 'bold', fontsize = 12)
    plt.legend(['a','b','c','d','e'])
    plt.grid(True)
    plt.show()


    
# -----------------------------------------------------------------------------------------------------------------------------------------
def both_boxplot(df, col, low_bound = 0 , up_bound = 1000000000, fig_size = (15,5)):
    """ Returns 2 or 3 boxplots of selected columns in selected dataframe"""
    """ First boxplot returns all values, including outliers"""
    """ If a lower or upper boundary is given, second boxplot narrows the first one between these boundaries"""
    """ Third boxplot excludes the outliers"""
    
    plt.figure(figsize = fig_size)
    plt.suptitle("Distribution of values in column {} ".format(col), fontsize = 12, fontweight = 'bold' )
    
    # Boxplot with every value
    plt.subplot(131)
    plt.boxplot(df[df[col].isna() == False][col], showfliers = True)
    plt.title('All values')
    plt.grid(True)
    
    # If a boundary is specified, plot the second boxplot
    if (low_bound != 0) or (up_bound != 1000000000) :
        plt.subplot(132)
        plt.boxplot(
            df[(df[col] > low_bound) & (df[col] < up_bound) & (df[col].isna() == False)][col], 
            showfliers = True)
        plt.title('With restricted values')
        plt.grid(True)
        
        third_index = 133
    else :
        third_index = 132
                    
    # Boxplot excluding outliers
    plt.subplot(third_index)
    plt.boxplot(df[df[col].isna() == False][col], showfliers = False)
    plt.title('No outliers')
    plt.grid(True)
    
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------
def higher_values(df, col, up_bound, limit = 10):
    
    return(df[df[col] > up_bound][['product_name','categories', 'main_category',col]].sort_values(by = col, ascending = False).head(limit))

# ----------------------------------------------------------------------------------------------------------------------------------------
def plot_score(df, x_crit, y_crit):
    
    plt.figure(figsize = (15,5))

    grades = df[~df['nutriscore_grade'].isna()]['nutriscore_grade'].sort_values().unique()
    colors = ['green','lime','yellow','orange','red']
    
    for i in range(0,5):
        x_ = df[(~df['nutriscore_score'].isna()) & (df['nutriscore_grade'] == grades[i])][x_crit]
        y_ = df[(~df['nutriscore_score'].isna()) & (df['nutriscore_grade'] == grades[i])][y_crit]

        plt.scatter(x_, y_, marker = "+", c = colors[i])

    plt.xlabel(x_crit)
    plt.ylabel(y_crit)
    plt.legend(grades)
    # plt.gca().set_xlim(x_limit)
    plt.grid(True)
    
    plt.show()
    
# ----------------------------------------------------------------------------------------------------------------------------------------
def plot_percentile(df, title = "Values per percentile", max_p = 95):
    """ Plots the values at each percentile for columns of the given dataframe. Three graphs are given :
    - First includes all values in each columns
    - Second is limited to the 99-th percentile
    - Last graph is limited by the max_p-th percentile
    """
    
    plt.figure(figsize = (20,5))
    plt.suptitle(title, fontsize = 12, fontweight = 'bold')
    
    perc_lim = [1.01, 1.00, (max_p/100) + 0.01]
    title_list = ["Including all values","Up to the 99-th percentile", "Up to the {}-th percentile".format(max_p)]
    
    for i in [0,1,2]:

        plt.subplot(131 + i)
        plt.plot(df.quantile(np.arange(0.01, perc_lim[i], 0.01)))
        plt.legend(df.columns)
        plt.xlabel("Percentile", backgroundcolor = 'lightgrey', fontstyle = 'italic')
        plt.ylabel("Value", backgroundcolor = 'lightgrey', fontstyle = 'italic')
        plt.grid(True)
        plt.title(title_list[i])
        
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------
def df_excess_rates(df, limit):
    """ Returns a barplot showing for each column of a dataset df the percent of values exceeding a given limit """
    
    nb_columns = len(df.columns) 
    
    df_excess = pd.DataFrame(df.applymap(lambda x: x>100).sum()/df.count())
    
    if len(df_excess[df_excess[0] > 0]) > 0:
        df_excess[df_excess[0] > 0].plot.bar(figsize = (15, 5), title = "Percentage of values > {}".format(limit), rot = 45)
        plt.grid(True)
        plt.gca().legend().set_visible(False)
    else:
        print("No columns with values > {} ".format(limit))
        
# -----------------------------------------------------------------------------------------------------------------------------------------
def plot_corr(df, selection, criteria):
    """ Returns a heatmap showing the strongers correlations between the selected columns and all other columns of df.
    Only the correlation index above criteria are shown"""
    
    # Creation of a correlation matrix
    df_correlations = df.corr(method = 'pearson')
    
    # Showing only correlations index above given criteria
    df_correlations = df_correlations.applymap(lambda x: 0 if abs(x) < criteria else x).copy()
    
    # Restricting to correlations for the selected columns
    df_selected_corr = df_correlations[selection]
    df_selected_corr = df_selected_corr.drop(index = selection)
    df_selected_corr['sum'] = df_selected_corr.sum(axis = 1)
    df_selected_corr = df_selected_corr[df_selected_corr['sum'] != 0].drop(columns = 'sum')
    
    # Creation of the heatmap
    sns.heatmap(df_selected_corr, cmap = 'coolwarm')
    
    