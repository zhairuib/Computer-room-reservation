#pragma once
#include"Identity.h"
#include<iostream>
#include<string>
#include"orderFile.h"
#include<vector>
using namespace std;

class Teacher :public Indentity
{
public:
	//Ĭ�Ϲ���
	Teacher();

	//�вι���
	Teacher(int empId, string name, string pwd);


	//�˵�
	virtual void operMenu();

	//�鿴����ԤԼ
	void showAllOrder();

	//���ԤԼ
	void validOrder();

	//ְ����
	int m_EmpId;
};

