import pandas as pd

def clean_up_test_data():
    include_patents = False

    df1 = pd.read_csv('data/dirty_data/Technology.csv')
    df1.columns = ['TechID', 'Description']
    df1.set_index('TechID', inplace=True)
    df1.replace('\n', '', regex=True, inplace=True)
    df1.drop([77162, 94680, 114279, 114286, 83483, 122988, 97546],
             inplace=True)  # remove highlight sets that are way too big, not representable
    print("(1/5) Num of technologies: ", df1.shape[0])
    df1 = df1[df1.Description.str.contains("(?i)\*\*highlights:\*\*")]  # Here we only take the Priliminary highlights.
    print("(2/5) Num of technologies with highlights (removing non-preliminary): ", df1.shape[0])
    df1 = df1.Description.str.split("(?i)\*\*highlights:\*\*", expand=True)
    df1 = df1.replace('(?<=\d\d\d\d\d\d\)\*\*)(.*?)(?=:\*\*)(.*)', '', regex=True).replace(
        '(?<=\d\d\d\d\d\d\) \*\*)(.*?)(?=:\*\*)(.*)', '', regex=True)  # Remove everything from the next **Title**
    df1 = df1.replace('\*', '', regex=True)  # removes the remaining *

    # I INCLUDED BOTH PATENTS AND ARTICLES RIGHT NOW

    if (include_patents):
        df1 = df1.iloc[:, 1].str.split(r'(?<=\d\d-)(\w*?)(?=\))', expand=True)
    else:
        df1 = df1.iloc[:, 1].str.split("(?<=\d\d-)(\d*?)(?=\))", expand=True)

    df1 = df1.replace(r'\[ \\(.*)', '', regex=True).replace(r'\[\\(.*)', '', regex=True)
    df1 = df1.replace(r'(?i)\[\[Art\. #ARTNUM\]\](.*)', '', regex=True).replace(r'(?i)art\. \[#ARTNUM\](.*)', '',
                                                                                regex=True).replace(r'\[#ARTNUM\](.*)',
                                                                                                    '', regex=True)
    df1 = df1.replace(r'\]\(#article(.*)', '', regex=True).replace(r'\(#article(.*)', '', regex=True)

    length = len(df1.columns) if (len(df1.columns) % 2 == 0) else len(df1.columns) - 1
    for column in range(length, -1, -2):
        if column != 0:
            df1.iloc[:, column] = df1.iloc[:, column].str[
                                  2:].str.lstrip().str.rstrip()  # delete the ') ' in front of the cells

    # This code makes a new df where the index is the same as the current code, and 1 column containing a dict with "highlight: articlenum" pairs
    new_list = []
    for row in range(len(df1)):
        new_dict = dict()
        length = len(df1.columns) - 2 if (len(df1.columns) % 2 == 0) else len(df1.columns) - 3
        for column in range(length, -1, -2):
            if (df1.iloc[row, column] != None and df1.iloc[row, column] != ""
                    and df1.iloc[row, column] != None and df1.iloc[row, column + 1] != ""):
                new_dict[df1.iloc[row, column]] = df1.iloc[row, column + 1]
        new_list.append(new_dict)

    new_df = pd.DataFrame({"highlights": new_list})
    new_df.set_index(df1.index, inplace=True)
    # This code splits the highlight (...) highlight instances and adds them to the dict, deleting the old instance

    for i, j in new_df.iterrows():
        temp_dict = j['highlights'].copy()
        for key, value in temp_dict.items():
            if '(...)' in str(key):
                subhighlights = str(key).split('(...)')
            else:
                subhighlights = str(key).split('\[...\]')

            # Remove empty strings. they are created when splitting on " (...) highlight"

            if len(subhighlights) > 1:
                if '' in subhighlights:
                    subhighlights.remove('')
                elif ' ' in subhighlights:
                    subhighlights.remove(' ')

                for sub_highlight in subhighlights:
                    sub_highlight = sub_highlight.lstrip().rstrip()

                    j['highlights'][sub_highlight] = value

                j['highlights'].pop(key)

    new_df.reset_index(level=0, inplace=True)



    tech_list = []
    high_list = []
    abs_list = []
    for i, j in new_df.iterrows():
        for key, value in j.highlights.items():
            tech_list.append(j.TechID)
            high_list.append(key.lstrip().rstrip())
            abs_list.append(value)
    # add the name of the the technology to the df
    df = pd.DataFrame({"TechID": tech_list, "Highlight": high_list, "AbstractID": abs_list})
    print('(3/6) Num of highlights: ', df.shape[0])
    drop_index = df[~df.AbstractID.notnull()].index
    df.drop(drop_index, inplace=True)  # DROP INSTANCES THAT DONT HAVE A ABSTRACTID, LIKE THE MISMATCH ON PATENTS ABOVE
    print('(4/6) Num of highlights with Abstract reference: ', df.shape[0])

    doc = 'data/dirty_data/TechnologyMagArticle.csv'
    dfabs = pd.read_csv(doc, error_bad_lines=False)  # , delimiter = ',')
    dfabs.columns = ['TechID', 'AbstractID', 'PaperAbstract']
    for i in range(1, 19):
        doc = 'data/dirty_data/TechnologyMagArticle (' + str(i) + ').csv'
        dftemp = pd.read_csv(doc, error_bad_lines=False)
        dftemp.columns = ['TechID', 'AbstractID', 'PaperAbstract']
        dfabs = pd.concat([dfabs, dftemp])

    print('(5/6) Num of presented abstracts: ', dfabs.shape[0])
    # we are missing some abstracts that belong to the highlight.

    dftech = pd.read_csv('data/dirty_data/Technology_Name.csv')
    dftech.columns = ["TechID", "TechName"]
    dfabs = pd.merge(dfabs, dftech, how="left", on="TechID")

    dfabs.AbstractID = dfabs.AbstractID.astype(str)
    df.AbstractID = df.AbstractID.astype(str)
    df = pd.merge(dfabs, df, how="left", on=["TechID", "AbstractID"])
    df.replace('\n', '', regex=True, inplace=True)

    # Not all technology names are present in the df, delete highlights that dont have a techname
    df = df[df.TechName.notnull()]

    print('(6/6) Num of presented abstracts with technology name present: ', df.shape[0])

    df.to_csv('data/test_set.csv')

def main():
    clean_up_test_data()


if __name__=='__main__':
    main()
