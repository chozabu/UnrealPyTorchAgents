// Fill out your copyright notice in the Description page of Project Settings.


#include "HelperFunLib.h"

#include "Runtime/Engine/Classes/PhysicsEngine/ConstraintInstance.h"
#include "Runtime/Engine/Classes/Animation/SkeletalMeshActor.h"
#include "Runtime/Engine/Classes/Components/SkeletalMeshComponent.h"

void UHelperFunLib::SetOrientationTarget(USkeletalMeshComponent* TargetMesh, FName ConstraintName, FRotator OrientationTarget) {
	auto constraint = TargetMesh->FindConstraintInstance(ConstraintName);
	constraint->SetAngularOrientationTarget(OrientationTarget.Quaternion());
}

FVector UHelperFunLib::GetSkelConstraintAngularForce(USkeletalMeshComponent* TargetMesh, FName ConstraintName) {
	auto constraint = TargetMesh->FindConstraintInstance(ConstraintName);
	FVector linforce(0, 0, 0);
	FVector angforce(0, 0, 0);
	constraint->GetConstraintForce(linforce, angforce);
	return angforce;
}

/*
void UHelperFunLib::SetOrientationTargetFromAction(USkeletalMeshComponent* TargetMesh, FName ConstraintName, FRotator OrientationTarget) {
	auto constraint = TargetMesh->FindConstraintInstance(ConstraintName);


	FRotator OrientationTargeta(0,0,0)
	constraint->SetAngularOrientationTarget(OrientationTarget.Quaternion());
}*/


FVector UHelperFunLib::GetOrientation(USkeletalMeshComponent* TargetMesh, FName ConstraintName) {

	FVector result(0, 0, 0);
	auto constraint = TargetMesh->FindConstraintInstance(ConstraintName);
	result.X = constraint->GetCurrentSwing1();
	result.Y = constraint->GetCurrentSwing2();
	result.Z = constraint->GetCurrentTwist();
	return result;
}

FVector UHelperFunLib::GetJointUnlocked(USkeletalMeshComponent* TargetMesh, FName ConstraintName) {

	FVector result(0, 0, 0);
	auto constraint = TargetMesh->FindConstraintInstance(ConstraintName);
	result.X = constraint->GetAngularTwistMotion() == EAngularConstraintMotion::ACM_Locked ? 0 : 1;
	result.Y = constraint->GetAngularSwing2Motion() == EAngularConstraintMotion::ACM_Locked ? 0 : 1;
	result.Z = constraint->GetAngularSwing1Motion() == EAngularConstraintMotion::ACM_Locked ? 0 : 1;
	return result;
}



TArray<FString> UHelperFunLib::GetConstraintNames(USkeletalMeshComponent* TargetMesh) {

	TArray<FString> names;
	for (auto c : TargetMesh->Constraints) {
		names.Add(c->JointName.ToString());
	}
	return names;
}


void UHelperFunLib::SetupJoint(USkeletalMeshComponent* TargetMesh, FName ConstraintName, float spring, float damp, float max_force) {
	auto constraint = TargetMesh->FindConstraintInstance(ConstraintName);
	constraint->SetAngularDriveMode(EAngularDriveMode::TwistAndSwing);
	constraint->SetOrientationDriveTwistAndSwing(true, true);
	constraint->SetAngularDriveParams(spring, damp, max_force);
	//constraint->angular
	//constraint->SetAngularDriveParams
}