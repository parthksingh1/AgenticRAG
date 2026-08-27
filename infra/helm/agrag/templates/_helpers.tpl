{{- define "agrag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agrag.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "agrag.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agrag.labels" -}}
app.kubernetes.io/name: {{ include "agrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{/*
The image tag must be set explicitly. Falling back to `latest` would let a
rescheduled pod silently run different code from the one it replaced.
*/}}
{{- define "agrag.image" -}}
{{- if not .Values.image.tag -}}
{{- fail "image.tag must be set to an immutable tag; `latest` is not acceptable" -}}
{{- end -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}

{{/*
Environment shared by the API, the workers and the beat scheduler, so a setting
added to one cannot be forgotten on the others.
*/}}
{{- define "agrag.env" -}}
- name: AGRAG_LOG_LEVEL
  value: {{ .Values.config.logLevel | quote }}
- name: AGRAG_ENVIRONMENT
  value: production
- name: AGRAG_DEFAULT_EMBEDDING_MODEL
  value: {{ .Values.config.embeddingModel | quote }}
- name: AGRAG_EMBEDDING_DIM
  value: {{ .Values.config.embeddingDim | quote }}
- name: AGRAG_RETRIEVAL_TOP_K
  value: {{ .Values.config.retrievalTopK | quote }}
- name: OTEL_SERVICE_NAME
  value: {{ include "agrag.fullname" . }}
- name: POD_NAME
  valueFrom:
    fieldRef: { fieldPath: metadata.name }
envFrom:
- secretRef:
    name: {{ .Values.secrets.existingSecret }}
{{- end -}}
