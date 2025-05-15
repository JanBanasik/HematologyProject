package com.agh.anemia.repository;

import com.agh.anemia.model.BloodTestResult;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BloodTestResultRepository extends JpaRepository<BloodTestResult, Long> {
}