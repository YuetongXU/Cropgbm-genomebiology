#!/usr/bin/perl
use strict;
use warnings;

my @vs=('9v1','7v1','5v1','3v1','1v1','1v3','1v5','1v7','1v9');
my @num=(69, 89, 124, 207, 621, 1863, 3105, 4347, 5589);

my @id;
open IN, "$ARGV[0]" or die $!;
my @data=<IN>;
close IN;

my $outdir=$ARGV[1];
`mkdir -p $outdir`;
for(my $i=0; $i<=$#num; $i++){
	my $n = $num[$i];
	my $v = $vs[$i];
	
	for(my $j = 1; $j<=30; $j++){
		open OUT, ">$outdir/r$v.m$j.id" or die $!;
		my %hash; 
		while ((keys %hash) < $n) { 
			$hash{int(rand(5589))} = 1;
		}
		foreach my $k(sort keys %hash){
			print OUT $data[$k];
		}
		close OUT;
	}
}
