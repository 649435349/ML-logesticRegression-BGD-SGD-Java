package WhatExcitingThings;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import Jama.Matrix;

public class zzzzzz {
	private static BufferedReader br;
	private static ArrayList<String> LineList = new ArrayList<String>();
	private static double alpha=0.000001;
	private static double[][] X,Y,W;
	private static Matrix MX,MY,MW,MWP;
	private static int row,column;
	
	public static ArrayList<String> GetList(File f) throws IOException//Get things needed from the file.
	{
		ArrayList<String> list1 = new ArrayList<String>();
	    try
	      {
	      FileReader fr=new FileReader(f);
	      br = new BufferedReader(fr);
	      try
	      {
	    	  String line = br.readLine();
	    	  while(line!=null)
	    	  {
	    		  	list1.add(line);
	    		  	line=br.readLine();
	    	  }
	      }
	      catch (IOException e)
	      {
	    	  e.printStackTrace();      
	      }
	      }
	      catch(FileNotFoundException e)
	      {
	    	  e.printStackTrace();
	      }
		return list1;
	}
	
	public static void InitMatrix()//Used to initialize the MatrixX , MatrixY and Matrix M.
	{
		row=LineList.size();
		String temp=LineList.get(0);
		String[] eee=temp.split(",");
		column=eee.length;
		X=new double[row][column];
		Y=new double[row][1];
		W=new double[1][column];
		for(int i=0;i<column;i++)
		{
			W[0][i]=1;
		}
		for(int i=0;i<row;i++)
		{
			X[i][0]=1;
		}
		for(int i=0;i<row;i++)
		{
		    temp=LineList.get(i);
			eee=temp.split(",");
			for(int j=0;j<column-1;j++)
			{
				X[i][j+1]=Double.parseDouble(eee[j]);
			}
			Y[i][0]=Double.parseDouble(eee[column-1]);
		}
		MX=new Matrix(X);
		MY=new Matrix(Y);
		MW=new Matrix(W);
	}
	
	public static Matrix sigmoid(Matrix M)//Function sigmoid
	{
		double[][] temp=new double[M.getRowDimension()][M.getColumnDimension()];
		for(int i=0;i<M.getRowDimension();i++)
		{
			for(int j=0;j<M.getColumnDimension();j++)
			{
				temp[i][j]=1/(1+Math.exp((-1)*M.get(i,j)));
			}
		}
		Matrix tempMat=new Matrix(temp);
		return tempMat;
	}
	
	public static void main(String args[]) throws IOException
	{
      int count=0;
	  Matrix M1,M2,M3;
      File f=new File("E:\\data.txt");
      LineList=GetList(f);
      InitMatrix();
      do
      {
    	  M1=MX.times(MW.transpose());
    	  M2=sigmoid(M1);
    	  M3=MY.minus(M2);
    	  MWP=MW.copy();
    	  MW=MW.plus((M3.transpose().times(MX)).times(alpha));
      }while(Math.abs((MWP.minus(MW)).get(0,0))>0.0000001 && Math.abs((MWP.minus(MW)).get(0,1))>0.00000001);
      System.out.print("W*X=");
      for(int i=0;i<column;i++)
      {
    	  System.out.print(MW.get(0,i));
      }
      for(int i=0;i<row;i++)
      {
    	  double result=0;
    	  for(int j=0;j<column;j++)
          {
    	     result+=X[i][j]*MW.get(0,j);
          }
    	  if((result>0 && Y[i][0]==1) || (result<0 && Y[i][0]==0) )
  	     {
  		    count++;//The amount of correct predict.
  		    //Well,the sample is so small that I just use the training ones.
  	     }
      }
      System.out.println(count);
	}
}
