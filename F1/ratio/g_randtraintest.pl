#!/usr/bin/perl
use strict;
use warnings;

my @vs=('1','2','3','4','5','6','7','8','9');

my @id;
open IN, "$ARGV[0]" or die $!;
my @data=<IN>;
close IN;

my $unit=69;
my @train;
my @test;
for(my $i=1; $i<=9; $i++){
	my $train = $unit*$i*9;
	my $test = $unit*$i;
	push @train, $train;
	push @test, $test;
}


my $outdir=$ARGV[1];
`mkdir -p $outdir`;
for(my $i=0; $i<=8; $i++){
	my $t1 = $train[$i];
	my $t2 = $test[$i];
	my $v = $vs[$i];
	
	for(my $j = 1; $j<=30; $j++){
		open OUT, ">$outdir/r$v.m$j.nine" or die $!;
		my %hash; 
		while ((keys %hash) < $t1) { 
			$hash{int(rand(6210))} = 1;
		}
		foreach my $k(sort keys %hash){
			print OUT $data[$k];
		}
		close OUT;
		
		open OUT, ">$outdir/r$v.m$j.one" or die $!;
		my %hash2;
		while ((keys %hash2) < $t2) {
			my $rn = int(rand(6210));
			unless (exists $hash{$rn}){
				$hash2{$rn} = 1;
			}
		}
		foreach my $k(sort keys %hash2){
			print OUT $data[$k];
		}
		close OUT;


	}
}
