#pragma once
#include"Identity.h"
#include<iostream>
#include<string>
#include"computerRoom.h"
#include<vector>
#include<fstream>
#include"globalFile.h" 
#include"orderFile.h"
using namespace std;

//ѧ����
class Student :public Indentity
{
public:
	//Ĭ�Ϲ���
	Student();
	//�вι���
	Student(int id, string name, string pwd);
	//�˵�����
	virtual void operMenu();

	//����ԤԼ
	void applyOrder();

	//�鿴����ԤԼ
	void showMyOrder();

	//�鿴����ԤԼ
	void showAllOrder();

	//ȡ��ԤԼ
	void cancelOrder();

	//ѧ��
	int m_Id;

	//��������
	vector<computerRoom>vCom;

};
