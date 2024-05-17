import MySQLdb

#---------追加------------
add_pt = ["タケルライコ","ミミッキュ","ハッサム","テツノツツミ","ランドロス"]
#---------------------------

#入力した文字列をdbに保存
def SQL_out(Pk_List,initia):
    # MySQLに接続する
    db = MySQLdb.connect(
        host='localhost',
        user='root',
        passwd='Seenyukkuri1',
        db='pokemondb',
        charset='utf8'
    )
    # カーソルを取得する
    cursor = db.cursor()

    if (initia == 1):
        cursor.execute("DROP TABLE IF EXISTS temochi")

        cursor.execute("CREATE TABLE temochi(pk_1 CHAR(10),pk_2 CHAR(10),pk_3 CHAR(10),pk_4 CHAR(10),pk_5 CHAR(10))")

    cursor.execute('INSERT INTO temochi (pk_1,pk_2,pk_3,pk_4,pk_5) VALUES (\'{}\',\'{}\',\'{}\',\'{}\',\'{}\');'.format(Pk_List[0],Pk_List[1],Pk_List[2],Pk_List[3],Pk_List[4],))

    # cursor.execute("LOAD DATA LOCAL INFILE 'pokemondata.csv' INTO TABLE temochi FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'")

    # cursor.execute("select * from temochi")

    # cursor.execute("SHOW columns from temochi")

    # for row in cursor:
    #     print(row)

    db.commit()

    # カーソルを閉じる
    cursor.close()

    # 接続を閉じる
    db.close()

def SQL_csv():
    # MySQLに接続する
    db = MySQLdb.connect(
        host='localhost',
        user='root',
        passwd='Seenyukkuri1',
        db='pokemondb',
        charset='utf8'
    )
    # カーソルを取得する
    cursor = db.cursor()
 
    #初期化
    cursor.execute("DROP TABLE IF EXISTS temochi")
    cursor.execute("DROP TABLE IF EXISTS Dragonite")

    #テーブルを作成
    cursor.execute("CREATE TABLE temochi(pk_1 CHAR(10),pk_2 CHAR(10),pk_3 CHAR(10),pk_4 CHAR(10),pk_5 CHAR(10))")
    cursor.execute("CREATE TABLE Dragonite(name CHAR(10),kata CHAR(10),tools CHAR(15))")

    #CVSをTableにインポート
    cursor.execute("LOAD DATA LOCAL INFILE 'pokemondata.csv' INTO TABLE temochi FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'")
    cursor.execute("LOAD DATA LOCAL INFILE 'Dragodata.csv' INTO TABLE Dragonite FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'")

    db.commit()

    # カーソルを閉じる
    cursor.close()

    # 接続を閉じる
    db.close()

    
# SQL_out(add_pt,0)
SQL_csv()