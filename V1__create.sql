create table siulibrary(Slid int not null, 
 lname varchar(20), 
 location varchar(20), 
 noofbranches int,
 primary key(Slid));



create table Ilibrary(Lid int not null,
lname varchar(20),
city varchar(20),
area varchar(20),
Slid int not null,
primary key(Lid),
foreign key(Slid) references siulibrary(Slid)); 