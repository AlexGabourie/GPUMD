/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <stdio.h>

struct AT_Para {
  double z[8];
  double rc2, rc6;
};

struct AT_Data {
  GPU_Vector<double> f12x;
  GPU_Vector<double> f12y;
  GPU_Vector<double> f12z;
};

class AT : public Potential
{
public:
  AT(FILE*, int num_of_types, const Neighbor& neighbor);
  virtual ~AT(void);
  virtual void compute(
    const int type_shift,
    const Box& box,
    const Neighbor& neighbor,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  void initialize_at_1(FILE*);
  void initialize_at_2(FILE*);
protected:
  AT_Para at_para;
  AT_Data at_data;
};
