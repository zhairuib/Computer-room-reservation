#pragma once
#include<iostream>
using namespace std;

//��ݳ������
class Indentity
{
public:

	//�����˵��������麯��
	virtual void operMenu() = 0;


	string m_UserName;
	string m_UserPassword;
};
