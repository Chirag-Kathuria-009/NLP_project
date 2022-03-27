create table Books(
Bid int not null,
Bname varchar(30),
Price int,
Lid int not null,
primary key(Bid),
foreign key(Lid) references Ilibrary(Lid));
