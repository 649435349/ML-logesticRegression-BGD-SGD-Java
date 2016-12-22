package WhatExcitingThings;
import java.io.*;
import java.util.*;
import Jama.*;

public class ExcitingThings 
{
	private static BufferedReader br;
	private static ArrayList<String> LineList = new ArrayList<String>();
	private static double alpha=0.001;
	private static double[][] X,Y,W;
	private static Matrix MX,MY,MW;
	private static int row,column;
	private static Matrix[] DivMX=new Matrix[4],DivMY=new Matrix[4];
	
	public static ArrayList<String> GetList(File f) throws IOException//Get lines needed from the file.
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
	
	public static void InitMatrix()//Used to initialize.4-fold.
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
		DivMX[0]=MX.getMatrix(0,row/4-1,0,column-1);
		DivMX[1]=MX.getMatrix(row/4,row/2-1,0,column-1);
		DivMX[2]=MX.getMatrix(row/2,3*row/4-1,0,column-1);
		DivMX[3]=MX.getMatrix(3*row/4,row-1,0,column-1);
		DivMY[0]=MY.getMatrix(0,row/4-1,0,0);
		DivMY[1]=MY.getMatrix(row/4,row/2-1,0,0);
		DivMY[2]=MY.getMatrix(row/2,3*row/4-1,0,0);
		DivMY[3]=MY.getMatrix(3*row/4,row-1,0,0);
		MW=new Matrix(W);
	}
	
	public static Matrix sigmoid(Matrix M)//Function sigmoid
	{
		double[][] temp=new double[M.getRowDimension()][M.getColumnDimension()];
		for(int i=0;i<M.getRowDimension();i++)
		{
			for(int j=0;j<M.getColumnDimension();j++)
			{
				temp[i][j]=Double.valueOf(1)/(1+Math.exp(-M.get(i,j)));
			}
		}
		Matrix tempMat=new Matrix(temp);
		return tempMat;
	}
	
	public static boolean Judge(Matrix MWP,Matrix MW)
	{
		int flag=0;
		for (int i =0;i<column;i++)
		{
			if(Math.abs(MWP.get(0,i)-MW.get(0, i))<0.00000001)flag++;
		}
		if(flag==column)return false;
		else return true;
	}
	
	public static void Outcome(Matrix TestMX,Matrix TestMY)//显示model结果和准确率
	{
		System.out.print("W*X="+MW.get(0,0)+"+");
	    for(int i=1;i<column;i++)
	    {
	    	System.out.print("("+MW.get(0,i)+")"+"*x"+i);
	    	if(i!=column-1)
	    	{
	    		System.out.print("+");
	    	}
	    }
	    System.out.println();
		int count=0;
	    for (int i=0;i<TestMX.getRowDimension();i++)
	    {
	    	double result=0;
	    	for (int j=0;j<TestMX.getColumnDimension();j++)
	    	{
	    		result+=TestMX.get(i, j)*MW.get(0, j);
	    	}
	    	if( (result>0 && TestMY.get(i, 0)==1)  || ( result<0 && TestMY.get(i, 0)==0))count++;
	    }
	    double accuracyrate=count/Double.valueOf(TestMX.getRowDimension())*100;
	    System.out.println("accuracyrate="+accuracyrate+"%");
	    System.out.println();
	}
	
	public static void BGD()//批梯度下降
	{
		System.out.println("---------------BGD----------------");
		InitMatrix();
		Matrix M1,M2,M3,MWP;
		for(int j=0;j<4;j++)
		{
		   for(int m=0;m<4;m++)
	       {
			    if(m!=j)
			    {
				//for (int i=0;i<times;i++)
			    do
			    {
	        	   M1=DivMX[m].times(MW.transpose());
	        	   M2=sigmoid(M1);
	               M3=DivMY[m].minus(M2);
	               MWP=MW.copy();
	    	       MW=MW.plus(((DivMX[m].transpose().times(M3)).times(alpha)).transpose());
			    }while(Judge(MWP,MW));
			    }
	       }
		   Outcome(DivMX[j],DivMY[j]);
		}
	}
	
	public static void SGD()//随机梯度下降
	{
		System.out.println("---------------SGD----------------");
		InitMatrix();
		Matrix M1,M2,M3,SubMX,SubMY,MWP;
		for(int j=0;j<4;j++)
		{
		   for(int m=0;m<4;m++)
	       {
			    if(m!=j)
			    {
		           for (int i=0;i<100000;i++)
		           {
	    	          Random r = new Random();
	    	          int rownum=r.nextInt(DivMX[m].getRowDimension()-1);
	    	          SubMX=DivMX[m].getMatrix(rownum,rownum,0,column-1);
	    	          SubMY=DivMY[m].getMatrix(rownum,rownum,0,0);   //随机取1行数据
	    	          M1=SubMX.times(MW.transpose());
	    	          M2=sigmoid(M1);
	    	          M3=SubMY.minus(M2);
	    	          MWP=MW.copy();
	    	          MW=MW.plus(((SubMX.transpose().times(M3)).times(alpha)).transpose());
		           } ;
			    }
	       }
		   Outcome(DivMX[j],DivMY[j]);
		}
	}
	
	public static void SubjectToL1L2()//加范数l1,l2
	{
		InitMatrix();
		Matrix M1,M2,M3,ML1=MW.copy(),ML2=MW.copy();
	    do
	    {
	    	double W1=0,W2=0;
	    	for (int i=0;i<column-1;i++)
			{
				W1+=Math.abs(MW.get(0,i));
				W2+=MW.get(0,i)*MW.get(0,i);
			}
			W2=Math.sqrt(W2);
	    	M1=MX.times(MW.transpose());
	    	M2=sigmoid(M1);
	        M3=MY.minus(M2);
	    	MWP=MW.copy();
	    	for(int i=0;i<column-1;i++)
	    	{
	    		ML1.set(0,i,W1/MW.get(0,i));
	    		ML2.set(0,i,W1/MW.get(0,i));
	    	}
	    	MW=(MW.plus((M3.transpose().times(MX)).times(alpha))).plus(ML1.plus(ML2));
	    }while(Math.abs((MWP.minus(MW)).get(0,0))>1);
	    System.out.print("W*X=");
	    for(int i=0;i<column-1;i++)
	    {
	    	System.out.print("("+MW.get(0,i)+")"+"*x"+i);
	    	if(i!=column-2)
	    	{
	    		System.out.print("+");
	    	}
	    }
	}
	
	public static void main(String args[]) throws IOException
	{
	  Scanner sc=new Scanner(System.in);
	  File f=new File("E:\\modifiediris.txt");
	  LineList=GetList(f);
	  Collections.shuffle(LineList);
      BGD();
      Outcome(MX,MY);//用所有数据集去测试准确率
      System.out.println();
      SGD();
      Outcome(MX,MY);//用所有数据集去测试准确率
      System.out.println();
      //SubjectToL1L2();
	}
}
