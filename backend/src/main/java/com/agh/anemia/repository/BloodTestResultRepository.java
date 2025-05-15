package com.agh.anemia.repository;

import com.agh.anemia.model.BloodTestResult;
import com.agh.anemia.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BloodTestResultRepository extends JpaRepository<BloodTestResult, Long> {
    Iterable<BloodTestResult> findByUser(User user);
}