ctiondef organize_data():
    """
    Load, organize, and normalize data
    """
    df = pd.read_csv('data/2017.csv')
    df.drop(['Arena', 'Rk', 'L', 'PL', 'PW', 'W', 'SOS', 'SRS', 'ORtg', 'DRtg', 'Attendance'], 1, inplace=True)
    df = df.set_index('Team')
    y = df['MOV']
    df = (df - df.mean()) / df.std()
    X = df.drop('MOV', 1)
    return X, y

if __name__ == '__main__':
    X, y = organize_data()
    iterative_forward_select(X, y, depth=5)


