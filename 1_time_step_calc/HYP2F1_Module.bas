
Attribute VB_Name = "Hypergeo2F1"
Option Explicit

' Gauss hypergeometric 2F1 with Euler transform + series summation.
' Usage in Excel: =HYP2F1(a, b, c, z, [tol], [nmax])
Public Function HYP2F1(a As Double, b As Double, c As Double, z As Double, _
                       Optional tol As Double = 1E-12, Optional nmax As Long = 200000) As Double
    Dim aa As Double, bb As Double, cc As Double, zz As Double
    Dim pref As Double, term As Double, s As Double, ratio As Double
    Dim n As Long

    If c = 0 Or c = -Int(Abs(c)) Then
        HYP2F1 = CVErr(xlErrNum)
        Exit Function
    End If

    If Abs(z) >= 0.7 Then
        zz = z / (z - 1#)
        aa = a
        bb = c - b
        cc = c
        pref = (1# - z) ^ (-a)
    Else
        zz = z
        aa = a
        bb = b
        cc = c
        pref = 1#
    End If

    s = 1#
    term = 1#
    For n = 1 To nmax
        ratio = ((aa + n - 1#) * (bb + n - 1#)) / ((cc + n - 1#) * n)
        term = term * ratio * zz
        s = s + term
        If Abs(term) < tol * Abs(s) Then Exit For
    Next n

    HYP2F1 = pref * s
End Function
