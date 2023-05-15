# File Operations in Python
Using Python as my primary language, but sometimes I really struggle with file handling. Usually, when I
read articles, books or just some kind of documents, I try to copy their codes to handle files. One of the reasons
is that in first, I didn't think it was important, because it was merely used to download, and open file.
But when I tried to make it by myself, it turned out something different and difficult as well. Their codes are really
complex in someways. Thus, in this post, I will try to learn how to handle files.

This post will cover:
- Open & Close
- Read & Write
- Copy, Move & Delete
- Search
- Move file pointer
- File Status
- Play with file path (the most terrifying stuff for me)

This post is inspired of a article in Toward Data Science, here is [link](https://towardsdatascience.com/knowing-these-you-can-cover-99-of-file-operations-in-python-84725d82c2df).

## Open & Close
Python has a built-in function open that opens the file and returns a file object. The type of the file object depends on the mode in which the file is opened. It can be a text file object, a raw binary file, and a buffered binary file. Every file object has methods such as read() and write().

```
file = open("test_file.txt","w+")
file.read()
file.write("a new line")
```

The most common modes are listed in the table. An important rule is that any w related mode will first truncate the file if it exists and then create a new file. Be careful with this mode if you don’t want to overwrite the file and use a append mode if possible.

| mode | meaning |
|r	| open for reading (default)|
|r+	| open for both reading and writing (file pointer is at the beginning of the file)|
|w	| open for writing (truncate the file if it exists)|
|w+	| open for both reading and writing (truncate the file if it exists)|
|a	| open for writing (append to the end of the file if exists & file pointer is at the end of the file)|

The problem in the previous code block is that we only opened the file, but didn’t close it. It’s important to always close the file when working with files. Having an open file object can cause unpredictable behaviors such as resource leak. There are two ways to make sure that a file is closed properly.

1. Use close() 
A good practice is to put it in finally, so that we can make sure the file will be closed in any case.

```
try:
    file = open('test_file.txt', 'w+')
    file.write('Hello')
exception Exception as e:
    logging.exception(e)
finally:
    file.close()
```

2. Use context manager with open(...) as f
With open() as f statement implements __enter__ and __exit__ methods to open and close the file. Besides, it encapsulates try/finally statement in the context manager, which means we will never forget to close the file.

```
with open('test_file', 'w+') as file:
    file.write('a new line')
```

## Read & Write
The file object provides 3 methods to read a file which are read(), readline() and readlines().
- read(size=-1) returns the entire contents of a file.
- readline(size=-1) returns an entire line including character \n at the end.
- readlines(hint=-1) returns all the lines of a file in a list. The optional parameter hint means if the number of characters returned exceeds hint, no more lines will be returned.

```
with open('test.txt', 'r') as reader:
    line = reader.readline()
    while line != "":
        line = reader.readline()
        print(line)
```

In terms of writing, there are 2 methods write() and writelines(). **It’s the responsibility of the developer to add \n at the end.**

```
with open('test.txt', 'w+') as f:
    f.write('hi\n')
    f.writelines(['this is a line\n', 'this is another line\n'])

#>>> cat test.txt
# hi
# this is a line
# this is another line
```

If you write text to a special file type such as JSON or csv, then you should use Python built-in module json or csv on top of file object.

```
import csv
import json

with open('cities.csv', 'w+') as file:
    writer = csv.DicWriter(file, fieldnames=['city', 'country'])
    writer.writerheader()
    writer.writerow({'city': 'Amsterdam', 'country': 'Netherlands'})
    writer.writerows(
        [
            {'city': 'Hanoi', 'country': 'Vietnam'},
            {'city': 'Melbourne', 'country': 'Australia'}
        ]
    )

with open('cities.json', 'w+') as file:
json.dump({'city': 'Amsterdam', 'country': 'Netherlands'}, file)

```

## Move pointer within the file
When we open a file, we get a file handler that points to a certain position. In r and w modes, the handler points to the beginning of the file. In a mode, the handler points to the end of the file.

As we read from the file, the pointer moves to the place where the next read will start from, unless we tell the pointer to move around. You can do this using 2 methods: tell() and seek().

tell() returns the current position of the pointer as number of bytes/characters from the beginning of the file. seek(offset,whence=0) moves the handler to a position offset characters away from whence. whence can be:

0: from the beginning of the file
1: from the current position
2: from the end of the file

## Understand the file status
The file system on the operating system can tell you a number of practical information about a file. For example, what’s the size of the file, when it was created and modified. To get this information in Python, you can use os or pathlib module. Actually there are many common things between os and pathlib. pathlib is a more object-oriented module than os.

***os***

A way to get a complete status is to use os.stat('test.txt').

```
print(os.stat("text.txt"))
>>> os.stat_result(st_mode=33188, st_ino=8618932538, st_dev=16777220, st_nlink=1, st_uid=501, st_gid=20, st_size=16, st_atime=1597527409, st_mtime=1597527409, st_ctime=1597527409)
```

You can get styatistics individually using os.path

```
os.path.getatime()
os.path.getctime()
os.path.getmtime()
os.path.getsize()
```

***Pathlib***

Another way to get the complete status is to use pathlib.Path("text.txt").stat(). It returns the same object as os.stat().

## Copy, Move and Delete a file

Python has many built-in modules to handle file movement. Before you trust the first answer returned by Google, you should be aware that different choices of modules can lead to different performances. Some modules will block the thread until the file movement is done, while others might do it asynchronously.

***shutil***

shutil is the most well-known module for moving, copying, and deleting both files and folders. It provides 4 methods to only copy a file. copy(), copy2() and copyfile().

copy() v.s. copy2(): copy2() is very much similar to copy(). The difference is that copy2() also copies the metadata of the file such as the most recent access time, the most recent modification time. But according to Python doc, even copy2() cannot copy all the metadata due to the constrain on the operating system.

```
shutil.copy("1.csv", "copy.csv")
shutil.copy2("1.csv", "copy2.csv")

print(pathlib.Path("1.csv").stat())
print(pathlib.Path("copy.csv").stat())
print(pathlib.Path("copy2.csv").stat())
# 1.csv
# os.stat_result(st_mode=33152, st_ino=8618884732, st_dev=16777220, st_nlink=1, st_uid=501, st_gid=20, st_size=11, st_atime=1597570395, st_mtime=1597259421, st_ctime=1597570360)

# copy.csv
# os.stat_result(st_mode=33152, st_ino=8618983930, st_dev=16777220, st_nlink=1, st_uid=501, st_gid=20, st_size=11, st_atime=1597570387, st_mtime=1597570395, st_ctime=1597570395)

# copy2.csv
# os.stat_result(st_mode=33152, st_ino=8618983989, st_dev=16777220, st_nlink=1, st_uid=501, st_gid=20, st_size=11, st_atime=1597570395, st_mtime=1597259421, st_ctime=1597570395)
```
copy() v.s. copyfile(): copy() sets the permission of the new file the same as the original file, but copyfile() doesn’t copy its permission mode. Secondly, the destination of copy() can be a directory. If a file with the same name exists, it will be overwritten, otherwise, a new file will be created. But, the destination of copyfile() must be the target file name.

***os***

os module has a function system() that allows you to execute the command in a subshell. You need to pass the command as an argument to the system(). This has the same effect as the command executed on the operating system. For moving and deleting files, you can also use dedicated functions in os module.

```
# copy
os.system("cp 1.csv copy.csv")

# rename/move
os.system("mv 1.csv move.csv")
os.rename("1.csv", "move.csv")

# delete
os.system("rm move.csv")
```

## Search a File
After copying and moving files, you will probably want to search for filenames that match a particular pattern. Python provides a number of built-in functions for you to choose from.

***glob***

>> The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell. It supports wildcard characters such as * ? [].

glob.glob("*.csv") searches for all the files that have csv extension in the current directory. glob module makes it possible to search for files in the subdirectories as well.

```
import glob
glob.glob('*.csv')
# ['1.csv', '2.csv']
glob.glob('**/*.csv', recursive=True)
# ['1.csv', '2.csv', 'source/3.csv']
```

***os***

We can simply list all the files in the directory using os.listdir() and use file.endswith() and file.startswith() to detect the pattern. If you want to traverse the directory, then use os.walk().

```
import os

for file in os.listdir('.'):
    if file.endswith('.csv'):
        print(file)

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            print(file)
```

***pathlib***

pathlib has a similar function to the glob module. It’s possible to search filenames recursively as well. Compared to the previous solution based on os, pathlib has less code and offers a more object-oriented solution.

```
from pathlib import Path
    
    p = Path('.')
    for name in p.glob('**/*.csv'):
        print(name)
```

## Play with file path

***relative and absolute path***

Both os and pathlib offer functions to get the relative path and absolute path of a file or a directory.

```
import os
import pathlib

print(os.path.abspath("1.txt"))
print(os.path.respath("1.txt"))

print(pathlib.Path('1.txt').absolute())
print(pathlib.Path('1.txt'))
```

***Joining Paths***

This is how we can join paths in os and pathlib independent of the environment. pathlib uses a slash to create child paths.

```
import os
import pathlib

print(os.path.join('/home', 'file.txt'))
print(pathlib.Path('/home') / 'file.txt')
```

***Getting the parent directory***

dirname() is the function to get parent directory in os, while in pathlib, you can just use Path().parent to get the parent folder.

```
import os
import pathlib

# relative path
print(os.path.dirname("source/2.csv"))
# source
print(pathlib.Path("source/2.csv").parent)
# source

# absolute path
print(pathlib.Path("source/2.csv").resolve().parent)
# /Users/<...>/project/source
print(os.path.dirname(os.path.abspath("source/2.csv")))
# /Users/<...>/project/source
```
