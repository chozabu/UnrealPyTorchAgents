// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "HelperFunLib.generated.h"

/**
 * 
 */
UCLASS()
class UNREALPYTORCHAGENTS_API UHelperFunLib : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category = "Torchey")
		static void SetOrientationTarget(USkeletalMeshComponent* TargetMesh, FName ConstraintName, FRotator OrientationTarget);

	UFUNCTION(BlueprintCallable, Category = "Torchey")
		static FVector GetOrientation(USkeletalMeshComponent* TargetMesh, FName ConstraintName);

	UFUNCTION(BlueprintCallable, Category = "Torchey")
		static TArray<FString> GetConstraintNames(USkeletalMeshComponent* TargetMesh);

	UFUNCTION(BlueprintCallable, Category = "Torchey")
		static void SetupJoint(USkeletalMeshComponent* TargetMesh, FName ConstraintName, float spring, float damp, float max_force);

	UFUNCTION(BlueprintCallable, Category = "Torchey")
		static FVector GetSkelConstraintAngularForce(USkeletalMeshComponent* TargetMesh, FName ConstraintName);

	UFUNCTION(BlueprintCallable, Category = "Torchey")
		static FVector GetJointUnlocked(USkeletalMeshComponent* TargetMesh, FName ConstraintName);

	
};
