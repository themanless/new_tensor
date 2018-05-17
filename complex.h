#ifndef COMPLEX_H
#define COMPLEX_H
#include <cufft.h>
#include <iostream>

class Complex
{
    double m_r;
    double m_i;

public:
    Complex()
    {
        m_r = 0.0;
        m_i = 0.0;
    }
    Complex(cufftComplex c)
    {
        m_r = c.x;
        m_i = c.y;
    }
    friend Complex operator+(const Complex &c1, const Complex &c2);

    friend Complex operator*(const Complex &c1, const Complex &c2);

    friend std::ostream& operator<< (std::ostream &out, const Complex &c);
};

Complex operator+(const Complex &c1, const Complex &c2)
{
    Complex c3;
    c3.m_r = c1.m_r + c2.m_r;
    c3.m_i = c1.m_i + c2.m_i;
    return c3;
}
Complex operator*(const Complex &c1, const Complex &c2)
{
    Complex c3;
    c3.m_r = c1.m_r*c2.m_r - c2.m_i*c1.m_i;
    c3.m_i = c1.m_r*c2.m_i + c2.m_r*c1.m_i;
    return c3;
}
std::ostream& operator<< (std::ostream &out, const Complex &c)
{
    out << c.m_r << "+" << c.m_i << "i";
    return out;
}

#endif
