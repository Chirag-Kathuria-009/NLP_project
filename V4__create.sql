create table newtable(
    Bnid int not null,
Bid int not null,
sample_id int,
Blid int,
editid int,
primary key(Bnid),
foreign key(Bid) references Books(Bid));

