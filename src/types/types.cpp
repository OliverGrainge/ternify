#include "types/types.h"

void free_QT(QT_S_I8_PT* qt) {
    if (qt) {
        delete[] qt->data;
        delete[] qt->scales;
        delete qt;
    }
}

void free_QT(QT_S_T_PT* qt) {
    if (qt) {
        delete[] qt->data;
        delete[] qt->scales;
        delete qt;
    }
}

void free_QT(QT_S_I8_PC* qt) {
    if (qt) {
        delete[] qt->data;
        delete[] qt->scales;
        delete qt;
    }
}

void free_QT(QT_S_T_PC* qt) {
    if (qt) {
        delete[] qt->data;
        delete[] qt->scales;
        delete qt;
    }
}

void free_QT(QT_S_I8_PG* qt) {
    if (qt) {
        delete[] qt->data;
        delete[] qt->scales;
        delete qt;
    }
}

void free_QT(QT_S_T_PG* qt) {
    if (qt) {
        delete[] qt->data;
        delete[] qt->scales;
        delete qt;
    }
}